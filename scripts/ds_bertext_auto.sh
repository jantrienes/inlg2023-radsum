#!/bin/bash
##########################################
# GSum (auto)
# 1. BertExt inference (train, valid, test)
# 2. Build dataset
# 3. GSum training + inference
##########################################
set -e
eval "$(conda shell.bash hook)"

STEP=$(cat $MODEL_BERTEXT/model_step_best.txt)
CHECKPOINT=$MODEL_BERTEXT/model_step_$STEP.pt
BERTEXT_SUMMARIES_PATH=$MODEL_BERTEXT/inference/

# Run BertExt inference on all splits
splits=(train valid test)
for split in ${splits[@]};
do
    outdir=$BERTEXT_SUMMARIES_PATH/$split/
    mkdir -p $outdir

    # BertExt Inference
    conda activate presumm
    python PreSumm/src/train.py \
      -task ext  \
      -mode test  \
      -test_split_name $split \
      -test_batch_size 5000  \
      -use_interval true  \
      -visible_gpus "${CUDA_VISIBLE_DEVICES:-1}"  \
      -max_pos 512  \
      -max_pred_sents 1 \
      -report_rouge false \
      -pretrained_model $PRETRAINED_MODEL  \
      -bert_data_path $DATA_UNGUIDED/bert/reports  \
      -log_file $outdir/inference.log  \
      -result_path $outdir/summaries  \
      -test_from $CHECKPOINT \
      -temp_dir temp/
done


####################################
# Build GSum data (auto)
conda activate guided-summary

for split in ${splits[@]};
do
   python -m guidedsum.guidance.extractive_guidance \
        --input_path $DATA_UNGUIDED/reports.$split.json \
        --output_path $DATA_Z_BERTEXT_AUTO/reports.$split.json \
        --z_ids_path $outdir/summaries.$STEP.candidate_ids_orig
done

python -m guidedsum.chunk --input_path $DATA_Z_BERTEXT_AUTO

conda activate presumm
python PreSumm/src/preprocess.py \
    -mode format_to_bert \
    -raw_path $DATA_Z_BERTEXT_AUTO/chunked/ \
    -save_path $DATA_Z_BERTEXT_AUTO/bert/ \
    -lower \
    -n_cpus 40 \
    -log_file logs/format_to_bert_$DATASET_NAME-oracle.log \
    -pretrained_model $PRETRAINED_MODEL \
    -min_src_nsents $MIN_SRC_NSENTS \
    -min_src_ntokens_per_sent $MIN_SRC_NTOKENS_PER_SENT \
    -min_tgt_ntokens $MIN_TGT_NTOKENS


####################################
# Build GSum data (auto+abstain)
conda activate guided-summary

# Use true abstain label during training, pred during validation and testing.
python -m guidedsum.guidance.extractive_guidance \
    --input_path $DATA_Z_BERTEXT_AUTO/reports.train.json \
    --output_path $DATA_Z_BERTEXT_AUTO_ABSTAIN/reports.train.json \
    --abstain_labels_path $MODEL_ABSTAIN_LR/y_true_train.txt \
    --z_abstain ""
python -m guidedsum.guidance.extractive_guidance \
    --input_path $DATA_Z_BERTEXT_AUTO/reports.valid.json \
    --output_path $DATA_Z_BERTEXT_AUTO_ABSTAIN/reports.valid.json \
    --abstain_labels_path $MODEL_ABSTAIN_LR/y_pred_valid.txt \
    --z_abstain ""
python -m guidedsum.guidance.extractive_guidance \
    --input_path $DATA_Z_BERTEXT_AUTO/reports.test.json \
    --output_path $DATA_Z_BERTEXT_AUTO_ABSTAIN/reports.test.json \
    --abstain_labels_path $MODEL_ABSTAIN_LR/y_pred_test.txt \
    --z_abstain ""


python -m guidedsum.chunk --input_path $DATA_Z_BERTEXT_AUTO_ABSTAIN

conda activate presumm
python PreSumm/src/preprocess.py \
    -mode format_to_bert \
    -raw_path $DATA_Z_BERTEXT_AUTO_ABSTAIN/chunked/ \
    -save_path $DATA_Z_BERTEXT_AUTO_ABSTAIN/bert/ \
    -lower \
    -n_cpus 40 \
    -log_file logs/format_to_bert_$DATASET_NAME-oracle.log \
    -pretrained_model $PRETRAINED_MODEL \
    -min_src_nsents $MIN_SRC_NSENTS \
    -min_src_ntokens_per_sent $MIN_SRC_NTOKENS_PER_SENT \
    -min_tgt_ntokens $MIN_TGT_NTOKENS
