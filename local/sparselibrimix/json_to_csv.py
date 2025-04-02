# Copyright (c) 2025 Nippon Telegraph and Telephone Corporation (NTT)
# Copyright (c) 2025 Brno University of Technology (BUT)
# This software is licensed under the NTT License (see LICENSE.NTT).

import argparse
import csv
import json
import os

import soundfile as sf


def json_to_csv(json_file, csv_file, mixture_path):
    with open(json_file, "r") as f:
        data = json.load(f)

    records = []

    for entry in data:
        record = {}
        record["mixture_ID"] = entry["mixture_name"]
        mixture_wavpath = os.path.join(mixture_path, "mix_noisy", f"{entry['mixture_name']}.wav")
        assert os.path.isfile(mixture_wavpath), f"Not exist: {mixture_wavpath}"
        record["mixture_path"] = mixture_wavpath

        src1 = os.path.join(mixture_path, "s1", f"{entry['mixture_name']}.wav")
        assert os.path.isfile(src1), f"Not exist: {src1}"
        record["source_1_path"] = src1

        src2 = os.path.join(mixture_path, "s2", f"{entry['mixture_name']}.wav")
        assert os.path.isfile(src2), f"Not exist: {src2}"
        record["source_2_path"] = src2

        noise = os.path.join(mixture_path, "noise", f"{entry['mixture_name']}.wav")
        assert os.path.isfile(noise), f"Not exist: {noise}"
        record["noise_path"] = noise

        record["length"] = sf.read(mixture_wavpath)[0].shape[0]
        records.append(record)

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON to CSV.")
    parser.add_argument("input_json", help="Path to input JSON file")
    parser.add_argument("output_csv", help="Path to output CSV file")
    parser.add_argument("--mixture", type=str, required=True, help="Path to mixture speech data")
    args = parser.parse_args()

    json_to_csv(args.input_json, args.output_csv, args.mixture)
