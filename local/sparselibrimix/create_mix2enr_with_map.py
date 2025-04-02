# Copyright (c) 2025 Nippon Telegraph and Telephone Corporation (NTT)
# Copyright (c) 2025 Brno University of Technology (BUT)
# This software is licensed under the NTT License (see LICENSE.NTT).

import argparse
import csv
import os

import soundfile as sf


def convert_list_to_csv(input_file, output_dir, mixture_path):
    data = []
    with open(input_file, "r") as f:
        for line in f:
            row = line.strip().split()
            enrwav = os.path.join(mixture_path, row[2]) + ".wav"
            wav_len = sf.read(enrwav)[0].shape[0]
            data.append([row[0], row[1], enrwav, wav_len])

    with open(os.path.join(output_dir, "mixture2enrollment.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mixture_id", "utterance_id", "enr_path1", "length1"])
        writer.writerows(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create mixture2enrollment file")
    parser.add_argument("map_file")
    parser.add_argument("output_dir")
    parser.add_argument("--mixture", type=str, required=True, help="Path to mixture speech data")
    args = parser.parse_args()

    convert_list_to_csv(args.map_file, args.output_dir, args.mixture)
