# TS-SUPERB

- TS-SUPERB includes four widely recognized target-speaker processing tasks that involve identifying the target speaker and extracting information from a speech mixture.
- TS-SUPERB serves as an extension of [S3PRL](https://github.com/s3prl/s3prl), a toolkit for SUPERB.

## Usage

### Extend the S3PRL Toolkit

First, extend the original S3PRL toolkit by copying files from this repository:

```shell
git clone https://github.com/s3prl/s3prl -b v0.4.17
# Install s3prl by following the instructions in its repository
git clone https://github.com/BUTSpeechFIT/TS_SUPERB
cd s3prl/s3prl
cp -r ../../TS_SUPERB/local ../../TS_SUPERB/downstream ../../TS_SUPERB/*.sh ./
```

### Data preparation

#### LibriMix

- Follow the instructions in the LibriMix repository:
  - https://github.com/JorisCos/LibriMix
- Enrollment speech is required to condition the model with the target speaker:
  - https://github.com/BUTSpeechFIT/speakerbeam
  - **Note**: The speakerbeam repository provides `speakerbeam/egs/libri2mix/local/prepare_data.sh`, which supports only 8kHz sampling and the `min` condition. Since this is insufficient for our requirements, we provide our own sample script to generate metadata for enrollment speech with 16kHz sampling and both `max` and `min` conditions: `TS-SUPERB/local/speakerbeam/prepare_data_min_max_16k.sh`. For example:
```shell
TS_SUPERB_PATH="path of TS_SUPERB"
LIBRIMIX_PATH="path of LibriMix"
git clone https://github.com/BUTSpeechFIT/speakerbeam
cd speakerbeam/egs/libri2mix
cp ${TS_SUPERB_PATH}/local/speakerbeam/prepare_data_min_max_16k.sh ./local/
./local/prepare_data_min_max_16k.sh ${LIBRIMIX_PATH}
# after processing, generate data/wav16k/{min/max}/{train-100/dev/test}/{mixture2enrollment/mixture_dev_mix_both/mixture_dev_mix_clean}.csv
```

#### SparseLibriMix

- For the test set, follow the instructions in the SparseLibriMix repository:
  - https://github.com/popcornell/SparseLibriMix
- For the training and validation sets with the corresponding enrollment speech, run the following command:

```shell
./local/prepare_data.sh <your_LibriSpeech_root_path> <your_WHAM_root_path> <your_SparseLibriMix_test_set_root_path>
```

### Running Target-Speaker Speech Processing Tasks

- For example, to run the TS-ASR experiment, use the following commands with the sample script:

```shell
cd s3prl/s3prl
./run_ts-asr.sh wavlm_base_plus <NGRAM_LEXICON_DIR>
# Here, <NGRAM_LEXICON_DIR> refers to the directory containing both `4-gram.arpa.gz` and `librispeech_lexicon.lst`, which can be downloaded from the links below:
# https://www.openslr.org/resources/11/4-gram.arpa.gz
# https://dl.fbaipublicfiles.com/fairseq/wav2vec/librispeech_lexicon.lst
# Decoding with the n-gram LM requires additional Python packages. Install them via the link below:
# https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/docs/superb.md#testing-with-kenlm--librispeech-official-4-gram-lm"
```

- You can run other TS tasks using the sample script `run_{pse/pvad/tse}.sh $UPSTREAM_MODEL_NAME`.
  - **Note**: For only the TS-ASR task, the default configuration uses `spk_conditioning_layer: 0`, which conditions the input of the main network with the target speaker embedding. This setting is different from the configuration used in the published paper, as it provides more stable training.
- (optional) The script `run_pvad_librimix.sh` runs a pVAD task using the LibriMix dataset.
  - This pVAD recipe depends on the `segments` file provided by the [original S3PRL speaker diarization recipe](https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/docs/superb.md#sd-speaker-diarization). Before running the script, you need to first complete the data preparation step of this recipe. Additionally, you must update the data directory paths (i.e., {train/dev/test}_dir) in `downstream/pvad/config_librimix.yaml` accordingly.

## Results

### TSE

- Scale-Invariant Signal-to-Distortion Ratio (SI-SDR), Short-Time Objective Intelligibility (STOI), and Perceptual Evaluation of Speech Quality (PESQ)

| Model       | SI-SDR | STOI  | PESQ  |
|:------------|-------:|------:|------:|
| HuBERT Base |  9.64  | 87.30 | 1.744 |
| WavLM Base  | 10.26  | 88.40 | 1.858 |
| WavLM Base+ | 10.69  | 89.00 | 1.915 |

### PSE

- Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) for each overlap ratio

| Model       | Ovl. 0% | Ovl. 20% | Ovl. 40% | Ovl. 60% | Average |
|:------------|--------:|---------:|---------:|---------:|--------:|
| HuBERT Base |  10.65  |   8.89   |   7.85   |   7.08   |   8.61  |
| WavLM Base  |  11.03  |  10.08   |   8.85   |   7.84   |   9.65  |
| WavLM Base+ |  11.94  |  10.55   |   9.22   |   8.33   |  10.01  |

- Short-Time Objective Intelligibility (STOI) for each overlap ratio

| Model       | Ovl. 0% | Ovl. 20% | Ovl. 40% | Ovl. 60% | Average |
|:------------|--------:|---------:|---------:|---------:|--------:|
| HuBERT Base |  86.10  |  81.20   |  77.50   |  74.90   |  79.92  |
| WavLM Base  |  87.30  |  82.90   |  79.40   |  76.70   |  81.57  |
| WavLM Base+ |  87.90  |  83.90   |  80.80   |  78.10   |  82.67  |

- Perceptual Evaluation of Speech Quality (PESQ) for each overlap ratio

| Model       | Ovl. 0% | Ovl. 20% | Ovl. 40% | Ovl. 60% | Average |
|:------------|--------:|---------:|---------:|---------:|--------:|
| HuBERT Base |  1.591  |  1.379   |  1.284   |  1.242   |  1.374  |
| WavLM Base  |  1.688  |  1.443   |  1.337   |  1.202   |  1.437  |
| WavLM Base+ |  1.737  |  1.480   |  1.372   |  1.313   |  1.475  |

### TS-ASR

- Word Error Rate (WER)

| Upstream Model | w/o LM | w/ LM |
|:---------------|-------:|------:|
| HuBERT Base    | 36.86  | 30.52 |
| WavLM Base     | 27.82  | 22.68 |
| WavLM Base+    | 24.75  | 20.06 |

### P-VAD

- mean Average Precision (mAP) for each overlap ratio

| Model       | Ovl. 0% | Ovl. 20% | Ovl. 40% | Ovl. 60% | Average |
|:------------|--------:|---------:|---------:|---------:|--------:|
| HuBERT Base |  94.25  |  94.75   |  94.74   |  94.66   |  94.60  |
| WavLM Base  |  94.09  |  94.57   |  94.78   |  94.35   |  94.40  |
| WavLM Base+ |  93.96  |  95.00   |  95.52   |  95.88   |  95.00  |

# Citation

If you find this repository helpful, please consider citing the following paper:

```text
@inproceedings{ts-superb,
  author={Junyi Peng and Takanori Ashihara and Marc Delcroix and Tsubasa Ochiai and Oldřich Plchot and Shoko Araki and Jan Černocký},
  title={{TS-SUPERB}: A Target Speech Processing Benchmark for Speech Self-Supervised Learning Models},
  year=2025,
  booktitle={IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)}
}
```