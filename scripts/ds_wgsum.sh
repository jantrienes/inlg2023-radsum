#!/bin/bash
eval "$(conda shell.bash hook)"
set -e

IN_PATH=$1
OUT_PATH=$IN_PATH/bert/
mkdir -p $OUT_PATH

conda activate wgsum

python WGSum/graph_construction/graph_construction.py $IN_PATH/train.jsonl
python WGSum/graph_construction/graph_construction.py $IN_PATH/valid.jsonl
python WGSum/graph_construction/graph_construction.py $IN_PATH/test.jsonl

python WGSum/src/preprocess.py \
    -mode format_to_bert \
    -raw_path $IN_PATH \
    -save_path $OUT_PATH  \
    -lower \
    -n_cpus 1 \
    -log_file $OUT_PATH/preprocess.log \
    -type edge_words \
    -min_src_nsents 1 \
    -min_src_ntokens_per_sent 3 \
    -min_tgt_ntokens 1
