#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate wgsum

mkdir -p $MODEL_WGSUM_CL temp/

python AIG_CL/src/train.py \
    -task abs \
    -mode train \
    -accum_count 5 \
    -batch_size 128 \
    -bert_data_path $DATA_WGSUM_CL/bert/radiology \
    -dec_dropout 0.2 \
    -log_file $MODEL_WGSUM_CL/training.log \
    -lr_bert 0.0002 \
    -lr_dec 0.05 \
    -model_path $MODEL_WGSUM_CL \
    -save_checkpoint_steps "${WGSUM_CL_CHECKPOINT_STEPS:-2000}" \
    -seed 777 \
    -sep_optim true \
    -train_steps "${WGSUM_CL_TRAIN_STEPS:-100000}" \
    -use_bert_emb true \
    -use_interval true \
    -warmup_steps_bert 10000 \
    -warmup_steps_dec 7000 \
    -visible_gpus "${CUDA_VISIBLE_DEVICES:-1,2,3}" \
    -max_pos 512 \
    -report_every 50 \
    -encoder bert \


python AIG_CL/src/train.py \
    -task abs \
    -mode validate \
    -batch_size 3000 \
    -test_batch_size 500 \
    -bert_data_path $DATA_WGSUM_CL/bert/radiology \
    -log_file $MODEL_WGSUM_CL/validate.log \
    -model_path $MODEL_WGSUM_CL \
    -sep_optim true \
    -use_interval true \
    -visible_gpus "${CUDA_VISIBLE_DEVICES:-0}" \
    -max_pos 512 \
    -max_length 200 \
    -alpha 0.95 \
    -min_length 5 \
    -result_path $MODEL_WGSUM_CL/summaries \
    -test_all \
    -metric_best "${GENERATION_BEST_METRIC:-ppl}"
