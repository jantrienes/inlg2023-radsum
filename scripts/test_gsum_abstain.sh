#!/bin/bash
eval "$(conda shell.bash hook)"
set -e
conda activate gsum-bert

GSUM_STEP=$(cat $MODEL_GSUM_ORACLE_ABSTAIN/model_step_best.txt)
CHECKPOINT=$MODEL_GSUM_ORACLE_ABSTAIN/model_step_$GSUM_STEP.pt

inference () {
  python GSum/bert/z_train.py \
    -task abs \
    -mode test \
    -batch_size 3000 \
    -test_batch_size 500 \
    -bert_data_path "$inferenceData" \
    -log_file "$logFile" \
    -sep_optim true \
    -use_interval true \
    -visible_gpus "${CUDA_VISIBLE_DEVICES:-1}" \
    -max_pos 512 \
    -max_length 200 \
    -alpha 0.95 \
    -min_length 5 \
    -result_path "$resultPath"/summaries \
    -test_from $CHECKPOINT \
    -pretrained_model $PRETRAINED_MODEL \
    -temp_dir temp/
}

datasets=(
  bertext-default-clip-k1-abstain
  bertext-default-clip-k2-abstain
  bertext-default-clip-k3-abstain
  bertext-default-clip-k4-abstain
  bertext-default-clip-k5-abstain
  bertext-default-clip-lrapprox-abstain
  bertext-default-clip-oracle-abstain
  bertext-default-clip-bertapprox-abstain
  bertext-default-clip-threshold-abstain
)

for dataset in ${datasets[@]}
do
  inferenceData=data/processed/$DATASET_NAME-$dataset/bert/reports
  resultPath=output/$DATASET_NAME-$dataset/gsum-default
  logFile=$resultPath/summaries.$GSUM_STEP.log

  if [[ -d "$resultPath" ]]
  then
    echo "$resultPath already exists. skipping..."
  else
    echo "======= run inference for $dataset ======="
    mkdir -p "$resultPath"
    inference
  fi
done
