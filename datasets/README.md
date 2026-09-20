# Datasets

This repository intentionally does not include large public surveillance datasets or raw CCTV footage. The folder exists for local dataset preparation, metadata, manifests, and tiny reference samples only.

## Purpose

The project uses several dataset families for training and evaluation, including:

- detection datasets for object and person recognition
- tracking datasets for multi-object association
- re-identification datasets for cross-camera identity matching
- anomaly and violence datasets for abnormal-event learning
- crowd-density datasets for density and flow estimation
- behavior datasets for loitering, counter-flow, and restricted-zone analysis
- incident and reasoning metadata for evaluation and narrative generation

## Local placement

Place downloaded datasets outside the Git tree in a local storage location, for example:

- /data/eagles-eye/datasets/
- C:/datasets/eagles-eye/
- /mnt/data/eagles-eye/

Then point the application configuration at the appropriate local path via the relevant environment variable or config setting.

## Required structure

The expected local structure is:

```text
datasets/
├── README.md
├── manifest.json
├── metadata/
│   ├── cameras.json
│   ├── zones.json
│   └── locations.json
├── detection/
│   ├── images/
│   ├── labels/
│   └── data.yaml
├── tracking/
├── reid/
├── anomaly/
├── behavior/
├── crowd/
├── objects/
├── occlusion/
├── appearance/
├── incidents/
└── raw/  # local, excluded from Git by default
```

## Official sources

- Open Images V7: official Google Open Images dataset
- XD-Violence: official violence dataset source
- UCF-Crime: official UCF-Crime dataset
- MOT17 / MOT20: official MOTChallenge releases
- ShanghaiTech Campus: official ShanghaiTech anomaly dataset
- UCF-QNRF: official crowd-counting dataset
- Market-1501: official Market-1501 person re-ID dataset

## Download and preprocessing

1. Review the license and terms for each dataset.
2. Download only the subset needed for the specific task.
3. Convert annotations to the project format required by the training scripts.
4. Validate with the project dataset tooling before any training run.
5. Keep only the generated metadata and any required labels under version control.

## Project modules using datasets

- detection: model training and evaluation
- tracking: object association and motion analysis
- reid: watchlist and cross-camera matching
- anomaly / violence: event classification and alerting
- behavior: loitering, counter-flow, following, and restricted-zone logic
- crowd: density, compression, and flow estimation
- incidents: reasoning and evidence aggregation

## Important release rule

Never commit raw video clips, dataset archives, multi-GB image sets, or training images to the Git repository. This repo is a source code and documentation release, not a data distribution package.
