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

expdir="exp/${upstream}/pvad_librimix"
mkdir -p "${expdir}"

echo "Starting training..."
python run_downstream.py \
  -m train \
  -p "${expdir}" \
  -u "${upstream}" \
  -d pvad \
  -c downstream/pvad/config_librimix.yaml \
  > "${expdir}/train.log" 2>&1

echo "Training completed. Log saved to ${expdir}/train.log"

echo "Starting evaluation..."
python run_downstream.py \
  -m evaluate \
  -e "${expdir}/best-states-dev.ckpt" \
  > "${expdir}/evaluate.log" 2>&1 
echo "Evaluation completed. Log saved to ${expdir}/evaluate.log"

# Summarize results
echo "tss_AP | mAP"
echo "--------------"
logfile="${expdir}/evaluate.log"
tss_ap=$(grep "test tss_AP:" "$logfile" | awk '{printf "%.3f", $3}')
map=$(grep "test mAP:" "$logfile" | awk '{printf "%.3f", $3}')
printf "%6s | %5s\n" "$tss_ap" "$map"