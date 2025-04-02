#!/bin/bash

# Copyright (c) 2025 Nippon Telegraph and Telephone Corporation (NTT)
# Copyright (c) 2025 Brno University of Technology (BUT)
# This software is licensed under the NTT License (see LICENSE.NTT).

set -e

if [ $# -lt 4 ]; then
    echo "Usage: $0 <LibriSpeech_root_path> <WHAM_root_path> <SparseLibriMix_test_root_path> <speakerbeam_root_path>"
    exit 1
fi

LIBRI_ROOT=$1
WHAM_ROOT=$2
SLM_TEST_ROOT=$3
SPKBEAM_ROOT=$4
RATE=16000

# Check required Python packages
REQUIREMENTS_URL="https://raw.githubusercontent.com/popcornell/SparseLibriMix/master/requirements.txt"
curl -sSf "$REQUIREMENTS_URL" -o local/requirements.txt || {
    echo "Failed to download requirements.txt"
    exit 1
}

MISSING_PACKAGES=()
while read -r line; do
    [[ "$line" =~ ^[[:space:]]*$ || "$line" =~ ^# ]] && continue
    PACKAGE=$(echo "$line" | cut -d '=' -f 1 | cut -d '<' -f 1 | cut -d '>' -f 1)
    python -c "import $PACKAGE" 2>/dev/null || MISSING_PACKAGES+=("$PACKAGE")
done < local/requirements.txt

if (( ${#MISSING_PACKAGES[@]} )); then
    echo "The following Python packages are missing:"
    for pkg in "${MISSING_PACKAGES[@]}"; do
        echo "- $pkg"
    done
    exit 1
fi

# Check and create symbolic link of each dataset
mkdir -p data
if [[ -d ${LIBRI_ROOT}/train-clean-100 && -d ${LIBRI_ROOT}/dev-clean && -d ${LIBRI_ROOT}/test-clean ]]; then
    if [ ! -L "data/librispeech" ]; then
        ln -sf ${LIBRI_ROOT} data/librispeech
    fi
else
    echo "Error: One or more LibriSpeech directories are missing."
    echo "Please download from: https://www.openslr.org/12"
    exit 1
fi

if [[ -d ${WHAM_ROOT}/wham_noise/tr && -d ${WHAM_ROOT}/wham_noise/cv && -d ${WHAM_ROOT}/wham_noise/tt ]]; then
    if [ ! -L "data/wham" ]; then
        ln -sf ${WHAM_ROOT} data/wham
    fi
else
    echo "Error: One or more WHAM directories are missing."
    echo "Please download from: http://wham.whisper.ai"
    exit 1
fi

if [[ -d ${SLM_TEST_ROOT}/sparse_2_0 && -d ${SLM_TEST_ROOT}/sparse_2_0.2 && \
      -d ${SLM_TEST_ROOT}/sparse_2_0.4 && -d ${SLM_TEST_ROOT}/sparse_2_0.6 ]]; then
    if [ ! -L "data/sparselibrimix_test" ]; then
        ln -sf ${SLM_TEST_ROOT} data/sparselibrimix_test
    fi
else
    echo "Error: One or more SparseLibriMix test-set directories are missing."
    echo "Follow the instructions at: https://github.com/popcornell/SparseLibriMix to create them."
    exit 1
fi

if [[ -d ${SPKBEAM_ROOT}/egs/libri2mix/data/wav16k/max/train-100 && \
      -d ${SPKBEAM_ROOT}/egs/libri2mix/data/wav16k/max/dev && \
      -d ${SPKBEAM_ROOT}/egs/libri2mix/data/wav16k/max/test && \
      -d ${SPKBEAM_ROOT}/egs/libri2mix/data/wav16k/min/train-100 && \
      -d ${SPKBEAM_ROOT}/egs/libri2mix/data/wav16k/min/dev && \
      -d ${SPKBEAM_ROOT}/egs/libri2mix/data/wav16k/min/test ]]; then
    if [ ! -L "data/speakerbeam" ]; then
        ln -sf ${SPKBEAM_ROOT} data/speakerbeam
    fi
else
    echo "Error: One or more speakerbeam metadata directories are missing."
    echo "Follow the instructions at: https://github.com/BUTSpeechFIT/speakerbeam to create them."
    exit 1
fi

# Create train and dev set of SparseLibriMix
if [ ! -d data/SparseLibriMix ]; then
    git clone https://github.com/popcornell/SparseLibriMix.git data/SparseLibriMix
fi

# Unarchive metadata.json
mdjson_dir="local/sparselibrimix/metadata/train-100"
if [ ! -f "${mdjson_dir}/metadata.json" ]; then
    tar -zxf ${mdjson_dir}/metadata.json.tar.gz -C ${mdjson_dir}
fi

mdjson_dir="local/sparselibrimix/metadata/dev"
if [ ! -f "${mdjson_dir}/metadata.json" ]; then
    tar -zxf ${mdjson_dir}/metadata.json.tar.gz -C ${mdjson_dir}
fi

# Create mixture speech
echo "Creating mixture speech..."
python data/SparseLibriMix/scripts/make_mixtures.py \
    local/sparselibrimix/metadata/train-100/metadata.json \
    data/librispeech/train-clean-100 data/sparselibrimix/train-100 \
    --noise_dir data/wham/wham_noise/tr --rate ${RATE}

python data/SparseLibriMix/scripts/make_mixtures.py \
    local/sparselibrimix/metadata/dev/metadata.json \
    data/librispeech/dev-clean data/sparselibrimix/dev \
    --noise_dir data/wham/wham_noise/cv --rate ${RATE}

# Convert JSON to CSV
echo "Converting metadata.json to CSV..."
python local/sparselibrimix/json_to_csv.py --mixture ${PWD}/data/sparselibrimix/train-100 \
    local/sparselibrimix/metadata/train-100/metadata.json \
    data/sparselibrimix/train-100/mixture_train-100_mix_both.csv

python local/sparselibrimix/json_to_csv.py --mixture ${PWD}/data/sparselibrimix/dev \
    local/sparselibrimix/metadata/dev/metadata.json \
    data/sparselibrimix/dev/mixture_dev_mix_both.csv

for ovr_ratio in 0 0.2 0.4 0.6; do
    mkdir -p data/sparselibrimix/test_2_${ovr_ratio}
    python local/sparselibrimix/json_to_csv.py --mixture data/sparselibrimix_test/sparse_2_${ovr_ratio}/wav16000 \
        data/SparseLibriMix/metadata/sparse_2_${ovr_ratio}/metadata.json \
        data/sparselibrimix/test_2_${ovr_ratio}/mixture_test_mix_both.csv
done

# Create symbolic links of metadata.json for segments information used in pvad task
ln -sf ../../../local/sparselibrimix/metadata/train-100/metadata.json data/sparselibrimix/train-100/
ln -sf ../../../local/sparselibrimix/metadata/dev/metadata.json data/sparselibrimix/dev/

for ovr_ratio in 0 0.2 0.4 0.6; do
    ln -sf ../../SparseLibriMix/metadata/sparse_2_${ovr_ratio}/metadata.json \
        data/sparselibrimix/test_2_${ovr_ratio}
done

# Create enrollment
echo "Creating enrollment data..."
python local/sparselibrimix/create_mix2enr.py --mixture ${PWD}/data/sparselibrimix/train-100 \
    local/sparselibrimix/metadata/train-100/metadata.json \
    data/sparselibrimix/train-100

python local/sparselibrimix/create_mix2enr_with_map.py --mixture ${PWD}/data/sparselibrimix/dev \
    local/sparselibrimix/metadata/dev/map_mixture2enrollment \
    data/sparselibrimix/dev

for ovr_ratio in 0 0.2 0.4 0.6; do
    python local/sparselibrimix/create_mix2enr_with_map.py --mixture data/sparselibrimix_test/sparse_2_${ovr_ratio}/wav16000 \
        local/sparselibrimix/metadata/test/map_mixture2enrollment \
        data/sparselibrimix/test_2_${ovr_ratio}
done

echo "Data prepration completed successfully."

