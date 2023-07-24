#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate guided-summary

MIMIC_RAW_PATH=data/raw/mimic
MIMIC_RAW_FILES=data/raw/mimic/files
MIMIC_INTERIM_PATH=data/interim/mimic
MIMIC_PROCESSED_PATH=data/processed/mimic
MIMIC_OFFICIAL_SPLIT_PROCESSED_PATH=data/processed/mimic-official

# Download MIMIC-CXR if it does not exist.
if [ -f $MIMIC_RAW_PATH/cxr-study-list.csv ];
then
    echo "Raw files exist. Skip download. To re-create, remove ${MIMIC_RAW_PATH}"
else
    wget -N -c -np -P ${MIMIC_RAW_PATH} --user $PHYSIONET_USER --ask-password \
        https://physionet.org/files/mimic-cxr/2.0.0/cxr-study-list.csv.gz \
        https://physionet.org/files/mimic-cxr/2.0.0/mimic-cxr-reports.zip \
        https://physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv.gz

    gunzip -c ${MIMIC_RAW_PATH}/cxr-study-list.csv.gz > ${MIMIC_RAW_PATH}/cxr-study-list.csv;
    gunzip -c ${MIMIC_RAW_PATH}/mimic-cxr-2.0.0-split.csv.gz > ${MIMIC_RAW_PATH}/mimic-cxr-2.0.0-split.csv;
    unzip -q -n ${MIMIC_RAW_PATH}/mimic-cxr-reports.zip -d ${MIMIC_RAW_PATH}
fi

if [ -f $MIMIC_INTERIM_PATH/mimic_cxr_sectioned.csv ];
then
    echo "Sectioned reports exist. Skip parsing. To re-create, remove ${MIMIC_INTERIM_PATH}"
else
    # Parse reports into sections
    python -m guidedsum.mimic.parse_sections \
        --reports_path $MIMIC_RAW_FILES \
        --output_path $MIMIC_INTERIM_PATH \
        --no_split
fi

if [ -f $MIMIC_PROCESSED_PATH/reports.train.json ];
then
    echo "Train/valid/test split exists. Skip preprocessing. To recreate, remove ${MIMIC_PROCESSED_PATH}"
else
    # Filter reports, get chexpert labels, split into train/valid/test
    python -m guidedsum.mimic.preprocess \
        --splits_csv_path $MIMIC_RAW_PATH/mimic-cxr-2.0.0-split.csv \
        --reports_path $MIMIC_INTERIM_PATH/mimic_cxr_sectioned.csv \
        --random_split_output_path $MIMIC_PROCESSED_PATH \
        --official_split_output_path $MIMIC_OFFICIAL_SPLIT_PROCESSED_PATH
fi

cmp data/external/mimic_train.txt $MIMIC_PROCESSED_PATH/train.ids
cmp data/external/mimic_valid.txt $MIMIC_PROCESSED_PATH/valid.ids
cmp data/external/mimic_test.txt $MIMIC_PROCESSED_PATH/test.ids
