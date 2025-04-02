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

expdir="exp/${upstream}/pse"
mkdir -p "${expdir}"

echo "Starting training..."
python run_downstream.py \
  -m train \
  -p "${expdir}" \
  -u "${upstream}" \
  -d pse \
  -c downstream/pse/config.yaml \
  > "${expdir}/train.log" 2>&1

echo "Training completed. Log saved to ${expdir}/train.log"

for ovr_ratio in 0 0.2 0.4 0.6; do
    echo "Evaluating with overlap ratio ${ovr_ratio}..."
    python run_downstream.py \
      -m evaluate \
      -e "${expdir}/best-states-dev.ckpt" \
      -o "config.downstream_expert.loaderrc.test_dir='data/sparselibrimix/test_2_${ovr_ratio}'" \
      > "${expdir}/evaluate_ovr${ovr_ratio}.log" 2>&1
    echo "Evaluation for overlap ratio ${ovr_ratio} completed. Log saved to ${expdir}/evaluate_ovr${ovr_ratio}.log"
done

# Summarize results
echo "Overlap Ratio | SI-SDR | STOI  | PESQ"
echo "----------------------------------------"
sum_si_sdr=0
sum_stoi=0
sum_pesq=0
for ovr_ratio in 0 0.2 0.4 0.6; do
    logfile=${expdir}/evaluate_ovr${ovr_ratio}.log
    si_sdr=$(grep "Average si_sdr" "$logfile" | awk '{printf "%.2f", $NF}')
    stoi=$(grep "Average stoi" "$logfile" | awk '{printf "%.3f", $NF}')
    stoi=$(echo "$stoi" | awk '{printf "%.2f\n", $1 * 100}')
    pesq=$(grep "Average pesq" "$logfile" | awk '{printf "%.3f", $NF}')
    sum_si_sdr=$(echo "$sum_si_sdr + $si_sdr" | bc)
    sum_stoi=$(echo "$sum_stoi + $stoi" | bc)
    sum_pesq=$(echo "$sum_pesq + $pesq" | bc)
    printf "%13s | %6s | %6s | %6s\n" "$ovr_ratio" "$si_sdr" "$stoi" "$pesq"
done
# Average
avg_si_sdr=$(echo "scale=2; $sum_si_sdr / 4" | bc)
avg_stoi=$(echo "scale=2; $sum_stoi / 4" | bc)
avg_pesq=$(echo "scale=3; $sum_pesq / 4" | bc)
avg_si_sdr=$(printf "%0.2f" "$avg_si_sdr")
avg_stoi=$(printf "%0.2f" "$avg_stoi")
avg_pesq=$(printf "%0.3f" "$avg_pesq")
echo "----------------------------------------"
printf "%13s | %6s | %6s | %6s\n" "Average" "$avg_si_sdr" "$avg_stoi" "$avg_pesq"
