#!/bin/bash
eval "$(conda shell.bash hook)"
set -e
conda activate gsum-bert

GSUM_STEP=$(cat $MODEL_GSUM_ORACLE/model_step_best.txt)
CHECKPOINT=$MODEL_GSUM_ORACLE/model_step_$GSUM_STEP.pt

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
  bertext-default-clip-k1
  bertext-default-clip-k2
  bertext-default-clip-k3
  bertext-default-clip-k4
  bertext-default-clip-k5
  bertext-default-clip-lrapprox
  bertext-default-clip-oracle
  bertext-default-clip-bertapprox
  bertext-default-clip-threshold
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
