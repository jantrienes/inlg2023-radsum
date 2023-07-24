#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate wgsum

mkdir -p $MODEL_WGSUM temp/

python WGSum/src/train.py \
    -mode train \
    -accum_count 5 \
    -batch_size 300 \
    -bert_data_path $DATA_WGSUM/bert/radiology \
    -log_file $MODEL_WGSUM/training.log \
    -lr 0.05 \
    -model_path $MODEL_WGSUM \
    -save_checkpoint_steps "${WGSUM_CHECKPOINT_STEPS:-2000}" \
    -seed 777 \
    -sep_optim false \
    -train_steps "${WGSUM_TRAIN_STEPS:-50000}" \
    -use_bert_emb true \
    -use_interval true \
    -warmup_steps 8000  \
    -visible_gpus "${CUDA_VISIBLE_DEVICES:-1,2,3,4}" \
    -max_pos 512 \
    -report_every 50 \
    -enc_hidden_size 512  \
    -enc_layers 6 \
    -enc_ff_size 2048 \
    -dec_dropout 0.1 \
    -enc_dropout 0.1 \
    -dec_layers 6 \
    -dec_hidden_size 512 \
    -dec_ff_size 2048 \
    -encoder baseline \
    -task abs

python WGSum/src/train.py \
    -task abs \
    -mode validate \
    -batch_size 3000 \
    -test_batch_size 500 \
    -bert_data_path $DATA_WGSUM/bert/radiology \
    -log_file $MODEL_WGSUM/validate.log \
    -model_path $MODEL_WGSUM \
    -sep_optim true \
    -use_interval true \
    -visible_gpus "${CUDA_VISIBLE_DEVICES:-0}" \
    -max_pos 512 \
    -max_length 200 \
    -alpha 0.95 \
    -min_length 5 \
    -result_path $MODEL_WGSUM/summaries \
    -test_all \
    -metric_best "${GENERATION_BEST_METRIC:-ppl}"
