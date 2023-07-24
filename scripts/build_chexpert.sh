#!/bin/bash
set -e

[[ ! -d chexpert-labeler ]] && git clone https://github.com/jantrienes/chexpert-labeler.git;
cd chexpert-labeler;
docker build -t chexpert-labeler:latest .
