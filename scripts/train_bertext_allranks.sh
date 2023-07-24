#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate presumm

DATA_PATH=$DATA_UNGUIDED/bert
MODEL_PATH=$MODEL_BERTEXT/allranks
mkdir -p $MODEL_PATH temp/

# Get best BertExt checkpoint
BERTEXT_STEP=$(cat $MODEL_BERTEXT/model_step_best.txt)
CHECKPOINT=$MODEL_BERTEXT/model_step_$BERTEXT_STEP.pt
BERTEXT_MAX_PRED_SENTS=15

python PreSumm/src/train.py \
    -task ext \
    -mode test \
    -test_batch_size 5000 \
    -bert_data_path $DATA_PATH/reports \
    -log_file $MODEL_PATH/test.log \
    -use_interval true \
    -visible_gpus "${CUDA_VISIBLE_DEVICES:-0}" \
    -max_pos 512 \
    -max_pred_sents $BERTEXT_MAX_PRED_SENTS \
    -result_path $MODEL_PATH/summaries \
    -test_from $CHECKPOINT \
    -pretrained_model $PRETRAINED_MODEL \
    -temp_dir temp/
