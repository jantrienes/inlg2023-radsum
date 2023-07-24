#!/bin/bash
eval "$(conda shell.bash hook)"

# Prepare (id, src, tgt)
conda activate guided-summary
python -m guidedsum.data_builder \
    --input_path $DATA_PROCESSED \
    --output_path $DATA_UNGUIDED \
    --dataset_name $DATASET_NAME \
    $DATA_BUILDER_EXTRA_ARGS

python -m guidedsum.chunk --input_path $DATA_UNGUIDED

# Bring data into PreSumm format
conda activate presumm

python PreSumm/src/preprocess.py \
    -mode format_to_bert \
    -raw_path $DATA_UNGUIDED/chunked/ \
    -save_path $DATA_UNGUIDED/bert/ \
    -lower \
    -n_cpus 40 \
    -log_file logs/format_to_bert_$DATASET_NAME-unguided.log \
    -pretrained_model $PRETRAINED_MODEL \
    -min_src_nsents $MIN_SRC_NSENTS \
    -min_src_ntokens_per_sent $MIN_SRC_NTOKENS_PER_SENT \
    -min_tgt_ntokens $MIN_TGT_NTOKENS
