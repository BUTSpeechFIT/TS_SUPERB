#!/bin/bash

# Copyright (c) 2025 Nippon Telegraph and Telephone Corporation (NTT)
# Copyright (c) 2025 Brno University of Technology (BUT)
# This software is licensed under the NTT License (see LICENSE.NTT).

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <upstream>"
    exit 1
fi

export PYTHONPATH=..:$PYTHONPATH
# https://github.com/mir-evaluation/mir_eval/issues/326
# https://github.com/numpy/numpy/issues/15764
export OMP_NUM_THREADS=1

upstream=$1

expdir="exp/${upstream}/tse"
mkdir -p "${expdir}"

echo "Starting training..."
python run_downstream.py \
  -m train \
  -p "${expdir}" \
  -u "${upstream}" \
  -d tse \
  -c downstream/tse/config.yaml \
  > "${expdir}/train.log" 2>&1

echo "Training completed. Log saved to ${expdir}/train.log"

echo "Starting evaluation..."
python run_downstream.py \
  -m evaluate \
  -e "${expdir}/best-states-dev.ckpt" \
  > "${expdir}/evaluate.log" 2>&1

echo "Evaluation completed. Log saved to ${expdir}/evaluate.log"

logfile="${expdir}/evaluate.log"
si_sdr=$(grep "Average si_sdr" "$logfile" | awk '{printf "%.2f", $NF}')
stoi=$(grep "Average stoi" "$logfile" | awk '{printf "%.3f", $NF}')
stoi=$(echo "$stoi" | awk '{printf "%.2f\n", $1 * 100}')
pesq=$(grep "Average pesq" "$logfile" | awk '{printf "%.3f", $NF}')
echo "SI-SDR | STOI  | PESQ"
echo "----------------------"
printf "%6s | %5s | %5s\n" "$si_sdr" "$stoi" "$pesq"
