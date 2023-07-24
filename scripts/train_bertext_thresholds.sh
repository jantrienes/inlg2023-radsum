#!/bin/bash
set -e
eval "$(conda shell.bash hook)"
conda activate presumm

SWEEP_OUTPUT=$MODEL_BERTEXT/threshold-sweep  # Path to store threshold sweep results
BEST_OUTPUT=$MODEL_BERTEXT_CLIPPED_THRESHOLD  # Path to store test set summaries with best threshold
CLIPPED_PATH=$DATA_Z_BERTEXT_CLIP_THRESHOLD  # Path to store GSum dataset with clipped summaries as guidance
mkdir -p $SWEEP_OUTPUT temp/

# Get best BertExt checkpoint
BERTEXT_STEP=$(cat $MODEL_BERTEXT/model_step_best.txt)
CHECKPOINT=$MODEL_BERTEXT/model_step_$BERTEXT_STEP.pt

# Step 1: run threshold sweep on validation set
python PreSumm/src/train.py \
  -task ext  \
  -mode test  \
  -test_split_name valid \
  -test_batch_size 5000  \
  -use_interval true  \
  -visible_gpus "${CUDA_VISIBLE_DEVICES:-1}"  \
  -max_pos 512  \
  -max_pred_sents 3 \
  -ext_threshold_sweep true \
  -report_rouge false \
  -pretrained_model $PRETRAINED_MODEL  \
  -bert_data_path $DATA_UNGUIDED/bert/reports  \
  -log_file $SWEEP_OUTPUT/threshold-validation.log  \
  -result_path $SWEEP_OUTPUT/summaries  \
  -test_from $CHECKPOINT  \
  -temp_dir temp/

# Step 2: evaluate summaries of each run and get best threshold
conda run -n guided-summary python scripts/evaluate_thresholds.py $SWEEP_OUTPUT $BERTEXT_STEP

# Step 3: use optimal threshold to generate summaries on the test set
BEST_THRESHOLD="`cat $SWEEP_OUTPUT/best.txt`"
echo "Running inference on test set with best threshold: $BEST_THRESHOLD"
mkdir -p $BEST_OUTPUT

python PreSumm/src/train.py \
  -task ext  \
  -mode test  \
  -test_batch_size 5000  \
  -use_interval true  \
  -visible_gpus "${CUDA_VISIBLE_DEVICES:-0}"  \
  -max_pos 512  \
  -max_pred_sents 3 \
  -report_rouge false \
  -pretrained_model $PRETRAINED_MODEL  \
  -bert_data_path $DATA_UNGUIDED/bert/reports  \
  -log_file $BEST_OUTPUT/summaries.$BERTEXT_STEP.log  \
  -ext_threshold $BEST_THRESHOLD \
  -result_path $BEST_OUTPUT/summaries  \
  -test_from "$CHECKPOINT"  \
  -temp_dir temp/

# Step 4: generate GSum dataset with BertExt (Thresholding) as guidance
mkdir -p $CLIPPED_PATH
cp $DATA_Z_ORACLE/reports.{train,valid}.json $CLIPPED_PATH
conda run -n guided-summary python -m guidedsum.guidance.extractive_guidance \
    --input_path $DATA_UNGUIDED/reports.test.json \
    --output_path $CLIPPED_PATH/reports.test.json \
    --z_ids $BEST_OUTPUT/summaries.$BERTEXT_STEP.candidate_ids_orig

conda run -n guided-summary python -m guidedsum.chunk \
  --input_path $CLIPPED_PATH

python PreSumm/src/preprocess.py \
  -mode format_to_bert \
  -raw_path $CLIPPED_PATH/chunked/ \
  -save_path $CLIPPED_PATH/bert/ \
  -lower \
  -n_cpus 40 \
  -log_file logs/format_to_bert_$DATASET_NAME-bertext-default-clip-threshold.log \
  -pretrained_model $PRETRAINED_MODEL \
  -min_src_nsents $MIN_SRC_NSENTS \
  -min_src_ntokens_per_sent $MIN_SRC_NTOKENS_PER_SENT \
  -min_tgt_ntokens $MIN_TGT_NTOKENS
