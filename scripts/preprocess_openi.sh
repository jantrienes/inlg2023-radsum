#!/bin/bash
# Download and preprocess OpenI.
# Raw data is fetched from: https://openi.nlm.nih.gov/faq#collection

eval "$(conda shell.bash hook)"
conda activate guided-summary

RAW_PATH=data/raw/openi
PROCESSED_PATH=data/processed/openi

if [ -d $RAW_PATH/ecgen-radiology/ ];
then
    echo "Raw files exist. Skip download. To re-create, remove ${RAW_PATH}"
else
    wget -P $RAW_PATH https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz
    tar -xzf $RAW_PATH/NLMCXR_reports.tgz -C $RAW_PATH
fi

if [ -f $PROCESSED_PATH/reports.train.json ];
then
    echo "Train/valid/test split exists. Skip preprocessing. To recreate, remove ${PROCESSED_PATH}"
else
    # Filter reports, get chexpert labels, split into train/valid/test
    python -m guidedsum.openi.preprocess \
        --reports_path $RAW_PATH/ecgen-radiology/ \
        --output_path $PROCESSED_PATH
fi

cmp data/external/openi_train.txt $PROCESSED_PATH/train.ids
cmp data/external/openi_valid.txt $PROCESSED_PATH/valid.ids
cmp data/external/openi_test.txt $PROCESSED_PATH/test.ids
