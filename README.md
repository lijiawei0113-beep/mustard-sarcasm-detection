# MUStARD Lightweight Multimodal Sarcasm Experiments

This folder contains reproducible code for the project proposal:

- Text-only baselines with TF-IDF and optional pre-extracted BERT.
- Early-fusion Text + Audio and Text + Audio + Vision classifiers.
- 5-fold cross-validation using MUStARD's released `split_indices.p`.
- SVM, Random Forest, and Logistic Regression comparisons.
- Plots and CSV tables for the report.

## Data already prepared

The small files needed for TF-IDF and audio experiments are in `data/`:

- `sarcasm_data.json`
- `split_indices.p`
- `audio_features.p`

These came from the public MUStARD repository.

## Run the lightweight experiments

```powershell
cd D:\Jiawei\Documents\mustard_experiment
python .\run_experiments.py
```

Outputs are written to `outputs/`:

- `summary_results.csv`
- `fold_results.csv`
- `weighted_f1_comparison.png`
- `metrics_heatmap.png`
- `fold_f1_lines.png`
- `best_confusion_matrix.png`
- `paper_svm_comparison.png`
- `audio_welch_ttests.csv`
- `audio_top_group_differences.png`

## Optional BERT and visual features

The original paper uses pre-extracted BERT and visual features. They are hosted on Hugging Face and are large:

- BERT zip: about 566 MB.
- Visual HDF5 files: about 2.6 GB total.

Download them only when needed:

```powershell
cd D:\Jiawei\Documents\mustard_experiment
python .\download_features.py --bert
python .\download_features.py --vision
python .\run_experiments.py
```

When BERT files are available, the script adds BERT text baselines. When the visual HDF5 files are available, it adds Text + Audio + Vision experiments.

## Notes for the report

The MUStARD paper reports SVM baselines, plus Majority and Random baselines. It does not report Random Forest in Table 2/3; Random Forest is included here as your own additional classical ML comparison.

The paper comparison plot uses the speaker-dependent 5-fold SVM scores from the original paper because this script uses the released 5-fold split file.
