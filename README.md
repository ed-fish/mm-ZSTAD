# Multi-modal Zero-shot Temporal Action Detection and Localization

This repository contains the code and implementation details for a series of experiments utilizing CLIP and CLAP for zero-shot temporal action localization. The primary goal is to explore the potential of multi-modal learning for detecting and localizing actions in videos without the need for training on extensive annotated data.

## Table of Contents

1. [Introduction](#introduction)
2. [Dependencies and Installation](#dependencies-and-installation)
3. [Dataset Preparation](#dataset-preparation)
4. [Usage Instructions](#usage-instructions)
5. [Experimental Results](#experimental-results)
6. [Citations](#citations)
7. [License](#license)

## Introduction

This project aims to leverage the powerful pre-trained models, CLIP (Contrastive Language-Image Pretraining) and CLAP (Contrastive Language-Audio Pretraining), to achieve zero-shot temporal action localization in videos. These models provide an interesting approach to learning from multi-modal data, such as image-text and audio-text pairs, and have demonstrated impressive zero-shot learning capabilities.

In our experiments, we explore different strategies to adapt these models for temporal action localization and detection tasks without the need for fine-tuning on specific action classes.

## Dependencies and Installation

To use this repository, you will need to have Python 3.7 or later installed. Additionally, you will need the following Python libraries:

- PyTorch
- torchvision
- torchaudio
- OpenCV
- NumPy
- Pandas

You can install the required packages using the following command:

```bash
pip install -r requirements.txt ```

## Dataset Preparation

Before running the experiments, you will need to prepare a dataset for temporal action localization. In this project, we used the Thumos 2014 dataset. Instructions for downloading and preparing the dataset can be found [here](DATASET.md).


## Usage Instructions

TBC

## Experimental Results

TBC

## Citations

TBC

## Licence

## License

This project is released under the [MIT License](LICENSE).

