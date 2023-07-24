#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate gsum-bert

DATA_PATH=$DATA_Z_ORACLE/bert
MODEL_PATH=$MODEL_GSUM_ORACLE
mkdir -p $MODEL_PATH temp/

python GSum/bert/z_train.py \
    -task abs \
    -mode train \
    -bert_data_path $DATA_PATH/reports \
    -dec_dropout 0.2 \
    -model_path $MODEL_PATH \
    -sep_optim true \
    -lr_bert $GSUM_LR_BERT \
    -lr_dec $GSUM_LR_DEC \
    -save_checkpoint_steps 2000 \
    -batch_size 140 \
    -train_steps $GSUM_TRAIN_STEPS \
    -report_every 50 \
    -accum_count 5 \
    -use_bert_emb true \
    -use_interval true \
    -warmup_steps_bert 20000 \
    -warmup_steps_dec 10000 \
    -max_pos 512 \
    -log_file $MODEL_PATH/train.log \
    -visible_gpus "${CUDA_VISIBLE_DEVICES:-1,2,3,4,5}" \
    -pretrained_model $PRETRAINED_MODEL \
    -temp_dir temp/

python GSum/bert/z_train.py \
    -task abs \
    -mode validate \
    -batch_size 3000 \
    -test_batch_size 1500 \
    -bert_data_path $DATA_PATH/reports \
    -log_file $MODEL_PATH/validate.log \
    -sep_optim true \
    -use_interval true \
    -visible_gpus "${CUDA_VISIBLE_DEVICES:-0}" \
    -max_pos 512 \
    -max_length 200 \
    -alpha 0.95 \
    -min_length $GSUM_MIN_LENGTH \
    -result_path $MODEL_PATH/summaries \
    -model_path $MODEL_PATH \
    -test_all True \
    -metric_best "${GENERATION_BEST_METRIC:-ppl}" \
    -pretrained_model $PRETRAINED_MODEL \
    -temp_dir temp/
