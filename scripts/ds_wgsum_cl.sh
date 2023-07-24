#!/bin/bash
eval "$(conda shell.bash hook)"
set -e

IN_PATH=$1
OUT_PATH=$2
mkdir -p $OUT_PATH $OUT_PATH/bert

conda activate wgsum

python AIG_CL/graph_construction/graph_construction.py $IN_PATH/train.jsonl $OUT_PATH/train_real_entity_with_graph.jsonl
python AIG_CL/graph_construction/graph_construction.py $IN_PATH/valid.jsonl $OUT_PATH/valid_real_entity_with_graph.jsonl
python AIG_CL/graph_construction/graph_construction.py $IN_PATH/test.jsonl $OUT_PATH/test_real_entity_with_graph.jsonl

python AIG_CL/src/preprocess.py \
    -mode format_to_bert \
    -raw_path $OUT_PATH \
    -save_path $OUT_PATH/bert/  \
    -lower \
    -n_cpus 1 \
    -log_file $OUT_PATH/bert/preprocess.log \
    -type edge_words \
    -min_src_nsents 1 \
    -min_src_ntokens_per_sent 3 \
    -min_tgt_ntokens 1
