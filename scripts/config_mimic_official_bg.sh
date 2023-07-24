#!/bin/bash

###### General ######
export DATASET_NAME=mimic-official-bg
export DATA_BUILDER_EXTRA_ARGS=--include_background


###### Model Hyperparameters ######
# PreSumm parameters
export PRETRAINED_MODEL=bert-base-uncased
export MIN_SRC_NSENTS=1
export MIN_SRC_NTOKENS_PER_SENT=3
export MIN_TGT_NTOKENS=1

# BertExt parameters
export BERTEXT_LR=0.002
export BERTEXT_TRAIN_STEPS=20000
export BERTEXT_WARMUP_STEPS=10000
export BERTEXT_MAX_PRED_SENTS=1

# BertAbs parameters
export BERTABS_LR_BERT=0.0002
export BERTABS_LR_DEC=0.02
export BERTABS_TRAIN_STEPS=20000
export BERTABS_MIN_LENGTH=5

# GSum parameters
export GSUM_LR_BERT=0.0002
export GSUM_LR_DEC=0.02
export GSUM_TRAIN_STEPS=20000
export GSUM_MIN_LENGTH=5

# BERT-Approx
export BERT_APPROX_CHECKPOINT=distilbert-base-cased


###### Path Configuration ######
export DATA_PROCESSED=data/processed/mimic-official  # We start from the default dataset, no -bg suffix needed here.
export DATA_UNGUIDED=data/processed/$DATASET_NAME-unguided
export DATA_Z_ORACLE=data/processed/$DATASET_NAME-oracle
export DATA_Z_ORACLE_ABSTAIN=data/processed/$DATASET_NAME-oracle-abstain
export DATA_Z_BERTEXT_CLIP_THRESHOLD=data/processed/$DATASET_NAME-bertext-default-clip-threshold
export DATA_Z_BERTEXT_AUTO=data/processed/$DATASET_NAME-bertext-auto
export DATA_Z_BERTEXT_AUTO_ABSTAIN=data/processed/$DATASET_NAME-bertext-auto-abstain
export DATA_WGSUM=data/processed/$DATASET_NAME-wgsum
export DATA_WGSUM_CL=data/processed/$DATASET_NAME-wgsum-cl

export MODEL_EXTORACLE=output/$DATASET_NAME-unguided/oracle
export MODEL_BERTEXT=output/$DATASET_NAME-unguided/bertext-default
export MODEL_BERTABS=output/$DATASET_NAME-unguided/bertabs-default
export MODEL_GSUM_ORACLE=output/$DATASET_NAME-oracle/gsum-default
export MODEL_GSUM_ORACLE_ABSTAIN=output/$DATASET_NAME-oracle-abstain/gsum-default
export MODEL_GSUM_BERTEXT_AUTO=output/$DATASET_NAME-bertext-auto/gsum-default
export MODEL_GSUM_BERTEXT_AUTO_ABSTAIN=output/$DATASET_NAME-bertext-auto-abstain/gsum-default
export MODEL_WGSUM=output/$DATASET_NAME-wgsum/wgsum-default
export MODEL_WGSUM_CL=output/$DATASET_NAME-wgsum-cl/wgsum-cl-default

export MODEL_BERTEXT_VARLEN_BASE_PATH=output/$DATASET_NAME-unguided/
export MODEL_BERTEXT_CLIPPED_THRESHOLD=output/$DATASET_NAME-unguided/bertext-default-clip-threshold
export MODEL_ORACLE_APPROX_LR=output/$DATASET_NAME-unguided/oracle_approx_lr
export MODEL_ORACLE_APPROX_BERT=output/$DATASET_NAME-unguided/oracle_approx_bert
export MODEL_ABSTAIN_LR=output/$DATASET_NAME-unguided/abstain_lr
