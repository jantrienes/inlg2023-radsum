#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate guided-summary

python -m guidedsum.guidance.abstain_lr \
    --data_path $DATA_UNGUIDED \
    --output_path $MODEL_ABSTAIN_LR

splits=(train valid test)
for split in ${splits[@]}
do
    python -m guidedsum.guidance.extractive_guidance \
        --input_path $DATA_Z_ORACLE/reports.$split.json \
        --output_path $DATA_Z_ORACLE_ABSTAIN/reports.$split.json \
        --abstain_labels_path $MODEL_ABSTAIN_LR/y_true_$split.txt \
        --z_abstain "No acute finding"
done

python -m guidedsum.chunk --input_path $DATA_Z_ORACLE_ABSTAIN

# Bring data into PreSumm format
conda activate presumm
python PreSumm/src/preprocess.py \
    -mode format_to_bert \
    -raw_path $DATA_Z_ORACLE_ABSTAIN/chunked/ \
    -save_path $DATA_Z_ORACLE_ABSTAIN/bert/ \
    -lower \
    -n_cpus 40 \
    -log_file logs/format_to_bert_$DATASET_NAME-oracle-abstain.log \
    -pretrained_model $PRETRAINED_MODEL \
    -min_src_nsents $MIN_SRC_NSENTS \
    -min_src_ntokens_per_sent $MIN_SRC_NTOKENS_PER_SENT \
    -min_tgt_ntokens $MIN_TGT_NTOKENS


auto_guidance=(
    bertext-default-clip-bertapprox
    bertext-default-clip-k1
    bertext-default-clip-k2
    bertext-default-clip-k3
    bertext-default-clip-k4
    bertext-default-clip-k5
    bertext-default-clip-lrapprox
    bertext-default-clip-oracle
    bertext-default-clip-threshold
)

for run in ${auto_guidance[@]}
do
    conda activate guided-summary
    outPath=data/processed/$DATASET_NAME-$run-abstain
    mkdir -p $outPath

    cp $DATA_Z_ORACLE_ABSTAIN/reports.{train,valid}.json $outPath
    python -m guidedsum.guidance.extractive_guidance \
        --input_path data/processed/$DATASET_NAME-$run/reports.test.json \
        --output_path $outPath/reports.test.json \
        --abstain_labels_path $MODEL_ABSTAIN_LR/y_pred_test.txt \
        --z_abstain "No acute finding"
    python -m guidedsum.chunk --input_path $outPath

    # Bring data into PreSumm format
    conda activate presumm
    python PreSumm/src/preprocess.py \
        -mode format_to_bert \
        -raw_path $outPath/chunked/ \
        -save_path $outPath/bert/ \
        -lower \
        -n_cpus 40 \
        -log_file logs/format_to_bert_$DATASET_NAME-$run-abstain.log \
        -pretrained_model $PRETRAINED_MODEL \
        -min_src_nsents $MIN_SRC_NSENTS \
        -min_src_ntokens_per_sent $MIN_SRC_NTOKENS_PER_SENT \
        -min_tgt_ntokens $MIN_TGT_NTOKENS
done
