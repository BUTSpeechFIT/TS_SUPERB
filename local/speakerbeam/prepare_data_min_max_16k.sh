#!/bin/bash

# Copyright (c) 2025 Nippon Telegraph and Telephone Corporation (NTT)
# Copyright (c) 2025 Brno University of Technology (BUT)
# Modifications made by NTT & BUT are licensed under the NTT License (see LICENSE.NTT).
# Original source: https://github.com/BUTSpeechFIT/speakerbeam/blob/main/egs/libri2mix/local/prepare_data.sh

. ../../path.sh

if [ $# -lt 1 ]; then
    echo 'One argument required (librimix_dir)'
    exit 1;
fi

librimix_dir=$1

# Overall metadata (by Asteroid recipes)
python local/create_local_metadata.py --librimix_dir $librimix_dir

# max-16k condition
# Enrollment utterances for test and dev
python local/create_enrollment_csv_fixed.py \
    data/wav16k/max/test/mixture_test_mix_both.csv \
    data/wav8k/min/test/map_mixture2enrollment \
    data/wav16k/max/test/mixture2enrollment.csv
python local/create_enrollment_csv_fixed.py \
    data/wav16k/max/dev/mixture_dev_mix_both.csv \
    data/wav8k/min/dev/map_mixture2enrollment \
    data/wav16k/max/dev/mixture2enrollment.csv

# Enrollment utterances for training
python local/create_enrollment_csv_all.py \
    data/wav16k/max/train-100/mixture_train-100_mix_both.csv \
    data/wav16k/max/train-100/mixture2enrollment.csv
# TS-SUPERB does not use train-360
# python local/create_enrollment_csv_all.py \
#     data/wav16k/max/train-360/mixture_train-360_mix_both.csv \
#     data/wav16k/max/train-360/mixture2enrollment.csv

# min-16k condition
# Enrollment utterances for test and dev
python local/create_enrollment_csv_fixed.py \
    data/wav16k/min/test/mixture_test_mix_both.csv \
    data/wav8k/min/test/map_mixture2enrollment \
    data/wav16k/min/test/mixture2enrollment.csv
python local/create_enrollment_csv_fixed.py \
    data/wav16k/min/dev/mixture_dev_mix_both.csv \
    data/wav8k/min/dev/map_mixture2enrollment \
    data/wav16k/min/dev/mixture2enrollment.csv

# Enrollment utterances for training
python local/create_enrollment_csv_all.py \
    data/wav16k/min/train-100/mixture_train-100_mix_both.csv \
    data/wav16k/min/train-100/mixture2enrollment.csv
# TS-SUPERB does not use train-360
# python local/create_enrollment_csv_all.py \
#     data/wav16k/min/train-360/mixture_train-360_mix_both.csv \
#     data/wav16k/min/train-360/mixture2enrollment.csv