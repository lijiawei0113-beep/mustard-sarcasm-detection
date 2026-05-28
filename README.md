# Multimodal Sarcasm Detection with MUStARD

This project investigates multimodal sarcasm detection using the MUStARD dataset. The goal is to evaluate whether combining textual, acoustic, and visual information improves sarcasm classification compared with text-only baselines.

Due to computational constraints, this project does not reproduce the GRU-based neural architectures from the original MUStARD paper. Instead, it uses pre-extracted multimodal features and lightweight machine learning classifiers implemented with scikit-learn.

## Project Overview

Sarcasm is often expressed not only through words, but also through tone of voice, facial expressions, gestures, and other non-verbal cues. This project compares three modality settings:

- Text only
- Text + Audio
- Text + Audio + Vision

For text representation, two approaches are used:

- TF-IDF
- BERT embeddings

For audio and vision, the project uses the pre-extracted features released with MUStARD:

- Audio: Librosa-based acoustic features
- Vision: ResNet-based visual features

The models are evaluated using the official five-fold cross-validation splits provided by MUStARD.

## Dataset

This project uses the Multimodal Sarcasm Detection Dataset, commonly known as MUStARD.

- Dataset: [soujanyaporia/MUStARD](https://github.com/soujanyaporia/MUStARD)
- Paper: [Towards Multimodal Sarcasm Detection: An Obviously Perfect Paper](https://aclanthology.org/P19-1455/)

MUStARD contains 690 utterances from television dialogues. Each utterance is labeled as sarcastic or non-sarcastic and includes text, audio, video, speaker information, and conversational context.

## Models

The following classical machine learning classifiers are used:

- Support Vector Machine (SVM)
- Random Forest (RF)
- Logistic Regression (LR)

SVM is included because it is directly comparable with the baseline reported in the original MUStARD paper. Random Forest and Logistic Regression are included as additional lightweight baselines.

## Feature Representations

### Text

Two text representations are used:

1. TF-IDF

   TF-IDF is used as a lightweight traditional text baseline. It represents utterances based on word and n-gram importance.

2. BERT

   Pre-extracted BERT-base-uncased embeddings are used as a stronger contextual text representation.

### Audio

Audio features are the pre-extracted Librosa-based acoustic features provided by MUStARD. These features include information related to MFCCs, Mel spectrograms, spectral centroid, and temporal derivatives.

### Vision

Visual features are pre-extracted ResNet-based representations from video frames. These features capture high-level visual information from the utterance videos.

## Feature Fusion

This project uses early fusion. Features from different modalities are concatenated before classification:

```text
Text
Text + Audio
Text + Audio + Vision
```

This allows direct comparison between unimodal and multimodal settings.

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1-score
- Sarcastic-class Precision
- Sarcastic-class Recall
- Sarcastic-class F1-score

Weighted F1-score is used as the main evaluation metric.

## Main Results

The best-performing model is:

```text
BERT + Audio + Vision + Logistic Regression
Weighted F1-score: 72.27
Accuracy: 72.32
```

The corresponding SVM model also performs strongly:

```text
BERT + Audio + Vision + SVM
Weighted F1-score: 72.01
```

These results are competitive with the SVM baselines reported in the original MUStARD paper.

## Repository Structure

```text
mustard_experiment/
├── run_experiments.py
├── download_features.py
├── requirements.txt
├── README.md
├── outputs/
│   ├── summary_results.csv
│   ├── fold_results.csv
│   ├── weighted_f1_comparison.png
│   ├── metrics_heatmap.png
│   ├── fold_f1_lines.png
│   ├── best_confusion_matrix.png
│   ├── paper_svm_comparison.png
│   ├── audio_top_group_differences.png
│   └── vision_top_group_differences.png
└── data/
    └── optional, not uploaded to GitHub
```

Large data files should not be committed to GitHub. They can be downloaded using `download_features.py`.

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/mustard-sarcasm-detection.git
cd mustard-sarcasm-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Setup

The smaller MUStARD files required for text and audio experiments should be placed under the `data/` directory:

```text
data/sarcasm_data.json
data/split_indices.p
data/audio_features.p
```

To download optional BERT and visual features:

```bash
python download_features.py --bert --vision
```

The optional feature files are large, so they should not be uploaded to GitHub.

## Run Experiments

Run the full experiment pipeline:

```bash
python run_experiments.py
```

The results will be saved to:

```text
outputs/
```

## Output Files

Important output files include:

- `summary_results.csv`: average results across five folds
- `fold_results.csv`: fold-level results
- `weighted_f1_comparison.png`: weighted F1 comparison across models
- `metrics_heatmap.png`: heatmap of accuracy, precision, recall, and F1
- `fold_f1_lines.png`: fold-wise F1 stability
- `best_confusion_matrix.png`: confusion matrix for the best model
- `paper_svm_comparison.png`: comparison with MUStARD paper SVM baselines
- `audio_top_group_differences.png`: Welch's t-test results for audio features
- `vision_top_group_differences.png`: Welch's t-test results for visual features

## Statistical Analysis

Welch's t-test is used to compare sarcastic and non-sarcastic utterances on each audio and visual feature dimension. This analysis examines whether non-textual modalities contain statistically meaningful differences between the two classes.

The audio and visual feature dimensions are indexed as:

```text
audio_000, audio_001, ...
vision_0000, vision_0001, ...
```

These indices refer to dimensions of the pre-extracted feature vectors, not directly interpretable human-labeled features.

## Notes

This project is a lightweight comparative study. It does not attempt to fully reproduce the original MUStARD neural architectures. Instead, it focuses on evaluating whether pre-extracted multimodal features combined with classical machine learning classifiers can provide competitive sarcasm detection performance.

## Citation

If you use MUStARD, please cite the original paper:

```bibtex
@inproceedings{castro2019towards,
  title={Towards Multimodal Sarcasm Detection (An Obviously Perfect Paper)},
  author={Castro, Santiago and Hazarika, Devamanyu and P{\'e}rez-Rosas, Ver{\'o}nica and Zimmermann, Roger and Mihalcea, Rada and Poria, Soujanya},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  pages={4619--4629},
  year={2019}
}
```
