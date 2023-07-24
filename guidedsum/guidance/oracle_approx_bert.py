"""Fine-tune DistilBERT for prediction of oracle length.

References:
- Fine-tuning based on: https://huggingface.co/docs/transformers/tasks/sequence_classification
- DistilBERT code: https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation
"""
import argparse
import tempfile
from pathlib import Path

import datasets
import evaluate
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from guidedsum.guidance.extractive_guidance import greedy_selection


def load_dataset(train_path, valid_path, test_path):
    # convert reports.train.json to reports.train.jsonl in a temporary directory
    # load as huggingface dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        print("created temporary directory", tmpdir)

        data_files = {}
        for split, in_path in zip(
            ["train", "valid", "test"], [train_path, valid_path, test_path]
        ):
            in_path = Path(in_path)
            out_path = tmpdir / f"{in_path.stem}.jsonl"
            df = pd.read_json(in_path)
            df.to_json(out_path, lines=True, orient="records")
            data_files[split] = str(out_path)

        ds = datasets.load_dataset("json", data_files=data_files)
    return ds


def predict(trainer, data, split_name: str, out_path):
    out_path = Path(out_path)
    preds = trainer.predict(data[split_name])
    y_pred = np.argmax(preds.predictions, axis=-1)
    y_true = preds.label_ids
    logger.info(f"Evaluate {split_name}\n" + classification_report(y_true, y_pred))

    with open(out_path / f"y_true_{split_name}.txt", "w") as fout:
        for i in y_true:
            fout.write(str(i) + "\n")

    with open(out_path / f"y_pred_{split_name}.txt", "w") as fout:
        for i in y_pred:
            fout.write(str(i) + "\n")


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def preprocess(row):
        """Add target (length of extractive oracle), and convert tokenized documents to string."""
        oracle_ids = greedy_selection(row["src"], row["tgt"], summary_size=3)
        oracle_len = len(oracle_ids)
        text = " ".join(sum(row["src"], []))
        return {"text": text, "label": oracle_len}

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True)

    data_path = Path(args.data_path)
    data = load_dataset(
        data_path / "reports.train.json",
        data_path / "reports.valid.json",
        data_path / "reports.test.json",
    )
    data = data.map(preprocess)
    data = data.map(tokenize, batched=True)

    labels = set(data["train"]["label"])
    num_labels = len(labels)
    logger.info(f"Number of labels = {num_labels} ({labels})")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=num_labels
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total:,}")
    logger.info(f"Trainable parameters: {trainable:,}")

    def compute_metrics(eval_pred):
        metric = evaluate.load("f1")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(
            predictions=predictions, references=labels, average="macro"
        )

    training_args = TrainingArguments(
        output_dir=args.output_path,
        num_train_epochs=3,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=512,
        logging_strategy="steps",
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=250,
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["valid"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    predict(trainer, data, split_name="train", out_path=args.output_path)
    predict(trainer, data, split_name="valid", out_path=args.output_path)
    predict(trainer, data, split_name="test", out_path=args.output_path)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")
    parser.add_argument("--output_path")
    parser.add_argument(
        "--base_model", help="Pre-trained huggingface model name or path."
    )
    return parser.parse_args()


if __name__ == "__main__":
    ARGS = arg_parser()
    logger.add(Path(ARGS.output_path) / "train.log")
    main(ARGS)
