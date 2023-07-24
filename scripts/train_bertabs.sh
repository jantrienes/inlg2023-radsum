#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate presumm

DATA_PATH=$DATA_UNGUIDED/bert
MODEL_PATH=$MODEL_BERTABS
mkdir -p $MODEL_PATH temp/

python PreSumm/src/train.py \
    -task abs \
    -mode train \
    -bert_data_path $DATA_PATH/reports \
    -dec_dropout 0.2 \
    -model_path $MODEL_PATH \
    -sep_optim true \
    -lr_bert $BERTABS_LR_BERT \
    -lr_dec $BERTABS_LR_DEC \
    -save_checkpoint_steps 2000 \
    -batch_size 140 \
    -train_steps $BERTABS_TRAIN_STEPS \
    -report_every 50 \
    -accum_count 5 \
    -use_bert_emb true \
    -use_interval true \
    -warmup_steps_bert 20000 \
    -warmup_steps_dec 10000 \
    -max_pos 512 \
    -log_file $MODEL_PATH/train.log \
    -visible_gpus "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}" \
    -pretrained_model $PRETRAINED_MODEL \
    -temp_dir temp/

python PreSumm/src/train.py \
    -task abs \
    -mode validate \
    -batch_size 3000 \
    -test_batch_size 500 \
    -bert_data_path $DATA_PATH/reports \
    -log_file $MODEL_PATH/validate.log \
    -sep_optim true \
    -use_interval true \
    -visible_gpus "${CUDA_VISIBLE_DEVICES:-0}" \
    -max_pos 512 \
    -max_length 200 \
    -alpha 0.95 \
    -min_length $BERTABS_MIN_LENGTH \
    -result_path $MODEL_PATH/summaries \
    -model_path $MODEL_PATH \
    -test_all True \
    -metric_best "${GENERATION_BEST_METRIC:-ppl}" \
    -pretrained_model $PRETRAINED_MODEL \
    -temp_dir temp/
