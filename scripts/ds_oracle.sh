#!/bin/bash
eval "$(conda shell.bash hook)"

conda activate guided-summary

splits=(train valid test)
for split in ${splits[@]}
do
    python -m guidedsum.guidance.extractive_guidance \
        --input_path $DATA_UNGUIDED/reports.$split.json \
        --output_path $DATA_Z_ORACLE/reports.$split.json \
        --use_oracle
done
python -m guidedsum.chunk --input_path $DATA_Z_ORACLE

# Bring data into PreSumm format
conda activate presumm
python PreSumm/src/preprocess.py \
    -mode format_to_bert \
    -raw_path $DATA_Z_ORACLE/chunked/ \
    -save_path $DATA_Z_ORACLE/bert/ \
    -lower \
    -n_cpus 40 \
    -log_file logs/format_to_bert_$DATASET_NAME-oracle.log \
    -pretrained_model $PRETRAINED_MODEL \
    -min_src_nsents $MIN_SRC_NSENTS \
    -min_src_ntokens_per_sent $MIN_SRC_NTOKENS_PER_SENT \
    -min_tgt_ntokens $MIN_TGT_NTOKENS
