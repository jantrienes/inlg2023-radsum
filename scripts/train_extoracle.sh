#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate presumm

DATA_PATH=$DATA_UNGUIDED/bert
MODEL_PATH=$MODEL_EXTORACLE

mkdir -p $MODEL_PATH temp/
python PreSumm/src/train.py \
    -task abs \
    -mode oracle \
    -bert_data_path $DATA_PATH/reports \
    -batch_size 3000 \
    -max_pos 512 \
    -use_interval true \
    -log_file $MODEL_PATH/train.log \
    -result_path $MODEL_PATH/summaries \
    -temp_dir temp/
