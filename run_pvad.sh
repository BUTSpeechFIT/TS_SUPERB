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

upstream=$1

expdir="exp/${upstream}/pvad"
mkdir -p "${expdir}"

echo "Starting training..."
python run_downstream.py \
  -m train \
  -p "${expdir}" \
  -u "${upstream}" \
  -d pvad \
  -c downstream/pvad/config.yaml \
  > "${expdir}/train.log" 2>&1

echo "Training completed. Log saved to ${expdir}/train.log"

echo "Starting evaluation..."
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
echo "Overlap Ratio | tss_AP | mAP"
echo "-------------------------------"
sum_tss=0
sum_map=0
for ovr_ratio in 0 0.2 0.4 0.6; do
    logfile=${expdir}/evaluate_ovr${ovr_ratio}.log
    tss_ap=$(grep "test tss_AP:" "$logfile" | awk '{printf "%.4f", $3}')
    map=$(grep "test mAP:" "$logfile" | awk '{printf "%.4f", $3}')
    sum_tss=$(echo "$sum_tss + $tss_ap" | bc)
    sum_map=$(echo "$sum_map + $map" | bc)
    printf "%13s | %6s | %6s\n" "$ovr_ratio" "$tss_ap" "$map"
done
# Average
avg_tss=$(echo "scale=3; $sum_tss / 4" | bc)
avg_map=$(echo "scale=3; $sum_map / 4" | bc)
avg_tss=$(printf "%0.4f" "$avg_tss")
avg_map=$(printf "%0.4f" "$avg_map")
echo "-------------------------------"
printf "%13s | %6s | %6s\n" "Average" "$avg_tss" "$avg_map"
