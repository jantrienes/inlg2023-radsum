#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate presumm

DATA_PATH=$DATA_UNGUIDED/bert
MODEL_PATH=$MODEL_BERTEXT
mkdir -p $MODEL_PATH temp/

python PreSumm/src/train.py \
    -task ext \
    -mode train \
    -ext_dropout 0.1 \
    -lr $BERTEXT_LR \
    -visible_gpus "${CUDA_VISIBLE_DEVICES:-1,2,3}" \
    -report_every 50 \
    -save_checkpoint_steps 1000 \
    -batch_size 3000 \
    -train_steps $BERTEXT_TRAIN_STEPS \
    -accum_count 2 \
    -use_interval true \
    -warmup_steps $BERTEXT_WARMUP_STEPS \
    -max_pos 512 \
    -pretrained_model $PRETRAINED_MODEL \
    -log_file $MODEL_PATH/train.log \
    -bert_data_path $DATA_PATH/reports \
    -model_path $MODEL_PATH \
    -temp_dir temp/


python PreSumm/src/train.py \
    -task ext \
    -mode validate \
    -batch_size 5000 \
    -test_batch_size 5000 \
    -bert_data_path $DATA_PATH/reports \
    -log_file $MODEL_PATH/validate.log \
    -use_interval true \
    -visible_gpus "${CUDA_VISIBLE_DEVICES:-0}" \
    -max_pos 512 \
    -result_path $MODEL_PATH/summaries \
    -model_path $MODEL_PATH \
    -test_all True \
    -metric_best "${GENERATION_BEST_METRIC:-ppl}" \
    -pretrained_model $PRETRAINED_MODEL \
    -max_pred_sents $BERTEXT_MAX_PRED_SENTS \
    -temp_dir temp/
