# Copyright (c) 2025 Nippon Telegraph and Telephone Corporation (NTT)
# Copyright (c) 2025 Brno University of Technology (BUT)
# This software is licensed under the NTT License (see LICENSE.NTT).

import argparse
import csv
import json
import os
from collections import defaultdict

import soundfile as sf


def parse_json(json_file, mixture):
    spk2utts = defaultdict(set)
    wavlen = {}
    with open(json_file, "r") as f:
        data = json.load(f)
        for entry in data:
            for spk_key in ["s1", "s2"]:
                spk = entry[spk_key][0]["file"].split("/")[0]
                mix_id = os.path.join(spk_key, entry["mixture_name"])
                utt_id = entry[spk_key][0]["file"].split("/")[-1].replace(".flac", "")
                spk2utts[spk].add((mix_id, utt_id))
                wavlen[mix_id] = sf.read(os.path.join(mixture, mix_id) + ".wav")[0].shape[0]
    return spk2utts, wavlen


def write_mixture2enrollment(json_file, output_dir, mixture):
    spk2utts, wavlen = parse_json(json_file, mixture)
    output_file = os.path.join(output_dir, "mixture2enrollment.csv")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mixture_id", "utterance_id", "enr_path1", "length1", "enr_path2", "length2", "..."])

        for entry in data:
            for tgt_spk in ["s1", "s2"]:
                spk = entry[tgt_spk][0]["file"].split("/")[0]
                tgt_utt = entry[tgt_spk][0]["file"].split("/")[-1].replace(".flac", "")

                mix_ids = [mix_id for mix_id, utt_id in spk2utts[spk] if tgt_utt != utt_id]

                row = [entry["mixture_name"], f"{tgt_spk}-{tgt_utt}"]
                for tgt_mix_id in mix_ids:
                    enrwav = os.path.join(mixture, tgt_mix_id) + ".wav"
                    # wav_len = sf.read(enrwav)[0].shape[0]
                    row.extend([enrwav, wavlen[tgt_mix_id]])

                writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create mixture2enrollment file")
    parser.add_argument("json_file", help="input json file")
    parser.add_argument("output_dir", help="output directory")
    parser.add_argument("--mixture", type=str, required=True, help="Path to mixture speech data")

    args = parser.parse_args()

    write_mixture2enrollment(args.json_file, args.output_dir, args.mixture)
