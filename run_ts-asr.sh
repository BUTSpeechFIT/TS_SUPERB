#!/bin/bash

# Copyright (c) 2025 Nippon Telegraph and Telephone Corporation (NTT)
# Copyright (c) 2025 Brno University of Technology (BUT)
# This software is licensed under the NTT License (see LICENSE.NTT).

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <upstream> <ngram_and_lexicon_path>"
    exit 1
fi

export PYTHONPATH=..:$PYTHONPATH

upstream=$1
ngram_lexicon_path=$2

# Check ngram and lexicon files
if [ ! -f ${ngram_lexicon_path}/4-gram.arpa.gz ]; then
    echo "Error: 4-gram.arpa.gz are missing."
    echo "Please download from: https://www.openslr.org/resources/11/4-gram.arpa.gz"
    exit 1
fi

if [ ! -f ${ngram_lexicon_path}/librispeech_lexicon.lst ]; then
    echo "Error: librispeech_lexicon.lst are missing."
    echo "Please download from: https://dl.fbaipublicfiles.com/fairseq/wav2vec/librispeech_lexicon.lst"
    exit 1
fi

# Check extra python package
extra_modules=("kenlm" "flashlight.lib.text" "flashlight.lib.sequence")
MISSING_PACKAGES=()
for PACKAGE in "${extra_modules[@]}"; do
    python -c "import $PACKAGE" 2>/dev/null || MISSING_PACKAGES+=("$PACKAGE")
done
if (( ${#MISSING_PACKAGES[@]} )); then
    echo "The following extra Python packages are missing:"
    for pkg in "${MISSING_PACKAGES[@]}"; do
        echo "- $pkg"
    done
    echo "Install them followed by https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/docs/superb.md#testing-with-kenlm--librispeech-official-4-gram-lm"
    exit 1
fi

expdir="exp/${upstream}/ts-asr"
mkdir -p "${expdir}"

echo "Starting training..."
python run_downstream.py \
  -m train \
  -p "${expdir}" \
  -u "${upstream}" \
  --seed 777 \
  -d ts-asr \
  -c downstream/ts-asr/config.yaml \
  > "${expdir}/train.log" 2>&1

echo "Training completed. Log saved to ${expdir}/train.log"

echo "Starting evaluation..."
python run_downstream.py \
  -m evaluate \
  -e "${expdir}/dev-best.ckpt" \
  > "${expdir}/evaluate.log" 2>&1

echo "Evaluation completed. Log saved to ${expdir}/evaluate.log"

echo "Starting evaluation with KenLM decoder..."
python run_downstream.py \
  -m evaluate \
  -e "${expdir}/dev-best.ckpt" \
  -o "\
        config.downstream_expert.datarc.decoder_args.decoder_type='kenlm',, \
        config.downstream_expert.datarc.decoder_args.kenlm_model='${ngram_lexicon_path}/4-gram.arpa.gz',, \
        config.downstream_expert.datarc.decoder_args.lexicon='${ngram_lexicon_path}/librispeech_lexicon.lst' \
     " \
  > "${expdir}/evaluate_lm.log" 2>&1

echo "Evaluation with KenLM decoder completed. Log saved to ${expdir}/evaluate_lm.log"

# Summarize results
echo "Condition | UER (%) | WER (%)"
echo "------------------------------"
for logfile in "${expdir}/evaluate.log" "${expdir}/evaluate_lm.log"; do
    filename=$(basename "$logfile")
    if [[ "$filename" == evaluate_lm.log ]]; then
        condition="w/  LM"
    else
        condition="w/o LM"
    fi
    uer=$(grep "test uer:" "$logfile" | awk '{printf "%.2f", $3}')
    wer=$(grep "test wer:" "$logfile" | awk '{printf "%.2f", $3}')
    printf "%-9s | %8s | %8s\n" "$condition" "$uer" "$wer"
done
