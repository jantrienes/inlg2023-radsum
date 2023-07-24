# MIMIC-CXR Error Analysis: Label Studio Deployment

This document describes how to setup [label-studio](https://labelstud.io/) for the error analysis, how to assign annotator tasks, and how to manage annotation files.

## Environment and deployment

```sh
conda create -n label-studio python=3.9
conda activate label-studio

pip install label-studio==1.6.0
```

Deployment on cluster node:

```sh
# For example on c51
screen -S label-studio

export LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=true
export LABEL_STUDIO_BASE_DATA_DIR=/local/work/jan/iccr-label-studio/
mkdir -p $LABEL_STUDIO_BASE_DATA_DIR

conda activate label-studio

# Setup admin User, and get the signup link from here "Organization > Add People"
# Then shutdown label studio.
label-studio start --username jan.trienes@uni-due.de --password [PASSWORD]

# Finally, start as usual.
label-studio start
```

Forwarding local port:

```sh
ssh c51 -L 8080:127.0.0.1:8080 -N
```

Initializing projects:

```sh
conda activate label-studio
export LABEL_STUDIO_USERNAME=jan.trienes@uni-due.de
export LABEL_STUDIO_PASSWORD=...
export LABEL_STUDIO_BASE_DATA_DIR=/local/work/jan/iccr-label-studio/
label-studio init "RRS: annotator1" --label-config error-analysis/label-studio-config.xml
label-studio init "RRS: annotator2" --label-config error-analysis/label-studio-config.xml
label-studio init "RRS: annotator3" --label-config error-analysis/label-studio-config.xml
label-studio init "RRS: annotator4" --label-config error-analysis/label-studio-config.xml
label-studio init "RRS: annotator5" --label-config error-analysis/label-studio-config.xml
label-studio init "RRS: annotator6" --label-config error-analysis/label-studio-config.xml
```

Setup periodic backups of database:

```sh
# ssh c51
mkdir -p /groups/dso/jan/iccr-backups/

# Start crontab editor
crontab -e

# Cronjob to backup Label Studio sqlite database (at minute 5 every 12 hours)
5 */12 * * * currentDate=`date "+\%Y\%m\%d-\%H\%M\%S"` && cd /local/work/jan/ && tar -zcf $currentDate.tar.gz iccr-label-studio/* && mv $currentDate.tar.gz /groups/dso/jan/iccr-backups/
```

## Assign tasks

1. Assign tasks in [error-analysis/data/assignments.xlsx](error-analysis/data/assignments.xlsx)
2. Generate label-studio task files with [notebooks/04-error-analysis-assignment.ipynb](notebooks/04-error-analysis-assignment.ipynb)
3. Open each annotator project and import task json.

## Exporting Annotations

```sh
conda activate label-studio
export LABEL_STUDIO_BASE_DATA_DIR=/local/work/jan/iccr-label-studio/

currentDate=`date "+%Y%m%d"`
label-studio export <PID> JSON_MIN --export-path=error-analysis/data/$currentDate-annotator1.json \
&& label-studio export <PID> JSON_MIN --export-path=error-analysis/data/$currentDate-annotator2.json \
&& label-studio export <PID> JSON_MIN --export-path=error-analysis/data/$currentDate-annotator3.json \
&& label-studio export <PID> JSON_MIN --export-path=error-analysis/data/$currentDate-annotator4.json \
&& label-studio export <PID> JSON_MIN --export-path=error-analysis/data/$currentDate-annotator5.json \
&& label-studio export <PID> JSON_MIN --export-path=error-analysis/data/$currentDate-annotator6.json
```

Sync remote with local copy:

```sh
rsync -auv -n c51:/groups/dso/jan/guided-summary/error-analysis/data error-analysis/data
```
