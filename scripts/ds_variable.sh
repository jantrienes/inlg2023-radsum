#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate guided-summary


if [ -f $MODEL_ORACLE_APPROX_LR/y_pred_test.txt ];
then
    echo "LR-Approx is already trained. Skip. To re-create, remove $MODEL_ORACLE_APPROX_LR/"
else
    python -m guidedsum.guidance.oracle_approx_lr \
        --data_path $DATA_UNGUIDED/ \
        --output_path $MODEL_ORACLE_APPROX_LR/
fi

if [ -f $MODEL_ORACLE_APPROX_BERT/y_pred_test.txt ];
then
    echo "BERT-Approx is already trained. Skip. To re-create, remove $MODEL_ORACLE_APPROX_BERT/"
else
    python -m guidedsum.guidance.oracle_approx_bert \
        --data_path $DATA_UNGUIDED/ \
        --output_path $MODEL_ORACLE_APPROX_BERT/ \
        --base_model $BERT_APPROX_CHECKPOINT
fi

# Generate BertExt runs with a variable output length
BERTEXT_STEP=$(cat $MODEL_BERTEXT/model_step_best.txt)
python -m guidedsum.guidance.bertext_clipped \
    --reports_json $DATA_UNGUIDED/reports.test.json \
    --bertext_ranks $MODEL_BERTEXT/allranks/ \
    --bertext_step $BERTEXT_STEP \
    --lr_approx_clips $MODEL_ORACLE_APPROX_LR/y_pred_test.txt \
    --bert_approx_clips $MODEL_ORACLE_APPROX_BERT/y_pred_test.txt \
    --out_path $MODEL_BERTEXT_VARLEN_BASE_PATH

# Generate GSum datasets with variable guidance
runs=(
  bertext-default-clip-k1
  bertext-default-clip-k2
  bertext-default-clip-k3
  bertext-default-clip-k4
  bertext-default-clip-k5
  bertext-default-clip-lrapprox
  bertext-default-clip-bertapprox
  bertext-default-clip-oracle
)

for run in ${runs[@]}
do
    outPath=data/processed/$DATASET_NAME-$run
    mkdir -p $outPath
    cp $DATA_Z_ORACLE/reports.{train,valid}.json $outPath
    python -m guidedsum.guidance.extractive_guidance \
        --input_path $DATA_UNGUIDED/reports.test.json \
        --output_path $outPath/reports.test.json \
        --z_ids output/$DATASET_NAME-unguided/$run/summaries.$BERTEXT_STEP.candidate_ids_orig

    python -m guidedsum.chunk --input_path $outPath/

    conda run -n presumm python PreSumm/src/preprocess.py \
        -mode format_to_bert \
        -raw_path $outPath/chunked/ \
        -save_path $outPath/bert/ \
        -lower \
        -n_cpus 40 \
        -log_file logs/format_to_bert_$DATASET_NAME-$run.log \
        -pretrained_model $PRETRAINED_MODEL \
        -min_src_nsents $MIN_SRC_NSENTS \
        -min_src_ntokens_per_sent $MIN_SRC_NTOKENS_PER_SENT \
        -min_tgt_ntokens $MIN_TGT_NTOKENS
done
