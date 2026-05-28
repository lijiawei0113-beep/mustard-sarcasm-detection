#!/usr/bin/env python
"""Lightweight MUStARD multimodal baselines.

Runs text, text+audio, and text+audio+vision experiments with 5-fold CV.
Text can be TF-IDF or the original pre-extracted BERT jsonl features.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.svm import SVC


PAPER_SPEAKER_DEPENDENT = [
    {"source": "MUStARD paper", "model": "Majority", "modality": "-", "precision": 25.0, "recall": 50.0, "f1": 33.3},
    {"source": "MUStARD paper", "model": "Random", "modality": "-", "precision": 49.5, "recall": 49.5, "f1": 49.8},
    {"source": "MUStARD paper", "model": "SVM", "modality": "T", "precision": 65.1, "recall": 64.6, "f1": 64.6},
    {"source": "MUStARD paper", "model": "SVM", "modality": "A", "precision": 65.9, "recall": 64.6, "f1": 64.6},
    {"source": "MUStARD paper", "model": "SVM", "modality": "V", "precision": 68.1, "recall": 67.4, "f1": 67.4},
    {"source": "MUStARD paper", "model": "SVM", "modality": "T+A", "precision": 66.6, "recall": 66.2, "f1": 66.2},
    {"source": "MUStARD paper", "model": "SVM", "modality": "T+V", "precision": 72.0, "recall": 71.6, "f1": 71.6},
    {"source": "MUStARD paper", "model": "SVM", "modality": "A+V", "precision": 66.2, "recall": 65.7, "f1": 65.7},
    {"source": "MUStARD paper", "model": "SVM", "modality": "T+A+V", "precision": 71.9, "recall": 71.4, "f1": 71.5},
]


@dataclass(frozen=True)
class Dataset:
    ids: list[str]
    texts: list[str]
    labels: np.ndarray
    shows: list[str]


def load_pickle(path: Path):
    with path.open("rb") as file:
        return pickle.load(file, encoding="latin1")


def load_dataset(data_dir: Path) -> Dataset:
    with (data_dir / "sarcasm_data.json").open(encoding="utf-8") as file:
        raw = json.load(file)
    ids = list(raw.keys())
    texts = [raw[id_]["utterance"] for id_ in ids]
    labels = np.array([int(raw[id_]["sarcasm"]) for id_ in ids], dtype=int)
    shows = [raw[id_].get("show", "") for id_ in ids]
    return Dataset(ids=ids, texts=texts, labels=labels, shows=shows)


def load_folds(data_dir: Path, labels: np.ndarray, n_splits: int = 5):
    split_path = data_dir / "split_indices.p"
    if split_path.exists():
        return load_pickle(split_path)
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(splitter.split(np.zeros(len(labels)), labels))


def pool_time_series(value: np.ndarray, pooling: str, time_axis: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        return arr
    if pooling == "mean":
        return np.nan_to_num(np.mean(arr, axis=time_axis))
    if pooling == "mean_std":
        return np.nan_to_num(np.concatenate([np.mean(arr, axis=time_axis), np.std(arr, axis=time_axis)]))
    raise ValueError(f"Unknown pooling: {pooling}")


def load_audio_features(data_dir: Path, ids: list[str], pooling: str) -> np.ndarray | None:
    path = data_dir / "audio_features.p"
    if not path.exists():
        return None
    raw = load_pickle(path)
    return np.vstack([pool_time_series(raw[id_], pooling=pooling, time_axis=1) for id_ in ids])


def load_vision_features(data_dir: Path, ids: list[str], pooling: str) -> np.ndarray | None:
    path = data_dir / "features" / "utterances_final" / "resnet_pool5.hdf5"
    if not path.exists():
        return None
    rows = []
    with h5py.File(path, "r") as file:
        for id_ in ids:
            rows.append(pool_time_series(file[id_][()], pooling=pooling, time_axis=0))
    return np.vstack(rows)


def load_bert_features(data_dir: Path, expected_rows: int) -> np.ndarray | None:
    npy_path = data_dir / "bert_features.npy"
    if npy_path.exists():
        features = np.load(npy_path)
        if len(features) != expected_rows:
            raise ValueError(f"{npy_path} has {len(features)} rows, expected {expected_rows}.")
        return features

    jsonl_path = data_dir / "bert-output.jsonl"
    if not jsonl_path.exists():
        return None

    rows = []
    with jsonl_path.open(encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            cls_token = item["features"][0]
            layers = cls_token["layers"]
            rows.append(np.mean([np.asarray(layer["values"], dtype=float) for layer in layers[:4]], axis=0))
    features = np.vstack(rows)
    if len(features) != expected_rows:
        raise ValueError(f"{jsonl_path} has {len(features)} rows, expected {expected_rows}.")
    np.save(npy_path, features)
    return features


def make_classifier(name: str, sparse_input: bool):
    if name == "svm":
        scaler = MaxAbsScaler() if sparse_input else StandardScaler()
        return make_pipeline(scaler, SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced"))
    if name == "logreg":
        scaler = MaxAbsScaler() if sparse_input else StandardScaler()
        return make_pipeline(
            scaler,
            LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=42),
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown classifier: {name}")


def combine_features(text_x, arrays: Iterable[np.ndarray], rows: np.ndarray):
    parts = [text_x]
    for arr in arrays:
        parts.append(sparse.csr_matrix(arr[rows]))
    return sparse.hstack(parts, format="csr")


def evaluate_fold(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100,
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0) * 100,
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100,
        "precision_sarcastic": precision_score(y_true, y_pred, pos_label=1, zero_division=0) * 100,
        "recall_sarcastic": recall_score(y_true, y_pred, pos_label=1, zero_division=0) * 100,
        "f1_sarcastic": f1_score(y_true, y_pred, pos_label=1, zero_division=0) * 100,
    }


def run_experiment(
    dataset: Dataset,
    folds,
    text_repr: str,
    modality: str,
    classifier_name: str,
    data_dir: Path,
    numeric_features: dict[str, np.ndarray],
    output_dir: Path,
    tfidf_max_features: int,
) -> tuple[list[dict[str, float]], np.ndarray]:
    fold_rows = []
    all_true = []
    all_pred = []
    y = dataset.labels

    bert = numeric_features.get("bert")
    if text_repr == "bert" and bert is None:
        raise RuntimeError("BERT features are not available.")

    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        if text_repr == "tfidf":
            vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=tfidf_max_features,
                sublinear_tf=True,
            )
            x_train_text = vectorizer.fit_transform([dataset.texts[i] for i in train_idx])
            x_test_text = vectorizer.transform([dataset.texts[i] for i in test_idx])
        elif text_repr == "bert":
            x_train_text = sparse.csr_matrix(bert[train_idx])
            x_test_text = sparse.csr_matrix(bert[test_idx])
        else:
            raise ValueError(f"Unknown text representation: {text_repr}")

        extra_names = [name for name in ["audio", "vision"] if name in modality]
        x_train = combine_features(x_train_text, [numeric_features[name] for name in extra_names], train_idx)
        x_test = combine_features(x_test_text, [numeric_features[name] for name in extra_names], test_idx)

        clf = make_classifier(classifier_name, sparse_input=True)
        clf.fit(x_train, y[train_idx])
        pred = clf.predict(x_test)

        metrics = evaluate_fold(y[test_idx], pred)
        metrics.update(
            {
                "fold": fold,
                "model": classifier_name,
                "text_repr": text_repr,
                "modality": modality.replace("audio", "A").replace("vision", "V").replace("text", "T"),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            }
        )
        fold_rows.append(metrics)
        all_true.extend(y[test_idx].tolist())
        all_pred.extend(pred.tolist())

    cm = confusion_matrix(all_true, all_pred, labels=[0, 1])
    return fold_rows, cm


def aggregate_results(fold_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "precision_sarcastic",
        "recall_sarcastic",
        "f1_sarcastic",
    ]
    group_cols = ["model", "text_repr", "modality"]
    mean = fold_df.groupby(group_cols)[metrics].mean().reset_index()
    std = fold_df.groupby(group_cols)[metrics].std(ddof=0).reset_index()
    std = std.rename(columns={col: f"{col}_std" for col in metrics})
    return mean.merge(std, on=group_cols)


def save_label_plot(dataset: Dataset, output_dir: Path) -> None:
    label_df = pd.DataFrame({"label": np.where(dataset.labels == 1, "Sarcastic", "Non-sarcastic"), "show": dataset.shows})
    plt.figure(figsize=(7, 4))
    sns.countplot(data=label_df, x="label", hue="label", palette=["#4C78A8", "#F58518"], legend=False)
    plt.title("MUStARD label balance")
    plt.xlabel("")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "label_balance.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    sns.countplot(data=label_df, x="show", hue="label", palette=["#4C78A8", "#F58518"])
    plt.title("Labels by TV show")
    plt.xlabel("Show")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "labels_by_show.png", dpi=220)
    plt.close()


def save_result_plots(agg_df: pd.DataFrame, fold_df: pd.DataFrame, cms: dict[str, np.ndarray], output_dir: Path) -> None:
    plot_df = agg_df.copy()
    plot_df["experiment"] = plot_df["text_repr"].str.upper() + " + " + plot_df["modality"] + " + " + plot_df["model"].str.upper()
    plot_df = plot_df.sort_values("f1_weighted", ascending=False)

    plt.figure(figsize=(11, max(4, 0.35 * len(plot_df))))
    sns.barplot(data=plot_df, y="experiment", x="f1_weighted", hue="model", dodge=False, palette="viridis")
    plt.xlim(0, 100)
    plt.title("Weighted F1 comparison")
    plt.xlabel("Weighted F1 (%)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "weighted_f1_comparison.png", dpi=220)
    plt.close()

    heat_df = plot_df.pivot_table(index="experiment", values=["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"])
    plt.figure(figsize=(8, max(4, 0.35 * len(heat_df))))
    sns.heatmap(heat_df, annot=True, fmt=".1f", cmap="YlGnBu", vmin=45, vmax=80)
    plt.title("Metric heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_heatmap.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5))
    line_df = fold_df.copy()
    line_df["experiment"] = line_df["text_repr"].str.upper() + " + " + line_df["modality"] + " + " + line_df["model"].str.upper()
    top = plot_df["experiment"].head(8).tolist()
    sns.lineplot(
        data=line_df[line_df["experiment"].isin(top)],
        x="fold",
        y="f1_weighted",
        hue="experiment",
        marker="o",
    )
    plt.ylim(0, 100)
    plt.title("Fold-wise weighted F1")
    plt.xlabel("Fold")
    plt.ylabel("Weighted F1 (%)")
    plt.tight_layout()
    plt.savefig(output_dir / "fold_f1_lines.png", dpi=220)
    plt.close()

    best_key = plot_df.iloc[0]["experiment"]
    cm = cms[best_key]
    plt.figure(figsize=(4.8, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-sarc", "Sarc"],
        yticklabels=["Non-sarc", "Sarc"],
    )
    plt.title(f"Best confusion matrix\n{best_key}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_dir / "best_confusion_matrix.png", dpi=220)
    plt.close()


def save_paper_comparison(agg_df: pd.DataFrame, output_dir: Path) -> None:
    paper = pd.DataFrame(PAPER_SPEAKER_DEPENDENT)
    paper.to_csv(output_dir / "paper_speaker_dependent_baselines.csv", index=False)

    ours = agg_df[agg_df["model"].eq("svm")].copy()
    if ours.empty:
        return
    ours["source"] = "This project"
    ours["f1"] = ours["f1_weighted"]
    ours["display"] = ours["source"] + " (" + ours["text_repr"].str.upper() + ")"
    ours = ours[["display", "modality", "f1"]]
    paper_svm = paper[paper["model"].eq("SVM")][["source", "modality", "f1"]].rename(columns={"source": "display"})
    compare = pd.concat([paper_svm, ours], ignore_index=True)
    compare = compare[compare["modality"].isin(["T", "T+A", "T+A+V"])]

    plt.figure(figsize=(8.5, 4.8))
    sns.barplot(data=compare, x="modality", y="f1", hue="display", palette=["#4C78A8", "#F58518", "#54A24B"])
    plt.ylim(0, 80)
    plt.title("Comparison with MUStARD paper SVM baselines")
    plt.xlabel("Modality")
    plt.ylabel("Weighted F1 (%)")
    plt.tight_layout()
    plt.savefig(output_dir / "paper_svm_comparison.png", dpi=220)
    plt.close()


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    n = len(p)
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        val = ranked[i] * n / (i + 1)
        prev = min(prev, val)
        adjusted[order[i]] = min(prev, 1.0)
    return adjusted


def save_feature_tests(name: str, features: np.ndarray, labels: np.ndarray, output_dir: Path) -> None:
    rows = []
    for col in range(features.shape[1]):
        pos = features[labels == 1, col]
        neg = features[labels == 0, col]
        stat, p_value = ttest_ind(pos, neg, equal_var=False, nan_policy="omit")
        rows.append(
            {
                "feature": f"{name}_{col:03d}",
                "mean_sarcastic": float(np.nanmean(pos)),
                "mean_non_sarcastic": float(np.nanmean(neg)),
                "difference": float(np.nanmean(pos) - np.nanmean(neg)),
                "t_stat": float(stat),
                "p_value": float(p_value),
            }
        )
    df = pd.DataFrame(rows)
    df["p_fdr"] = benjamini_hochberg(df["p_value"].fillna(1.0).to_numpy())
    df = df.sort_values("p_value")
    df.to_csv(output_dir / f"{name}_welch_ttests.csv", index=False)

    top = df.head(20).copy()
    top["minus_log10_p"] = -np.log10(top["p_value"].clip(lower=1e-300))
    plt.figure(figsize=(8, 6))
    sns.barplot(data=top, y="feature", x="minus_log10_p", hue="difference", palette="coolwarm", dodge=False)
    plt.title(f"Top {name} group differences")
    plt.xlabel("-log10(p), Welch t-test")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_top_group_differences.png", dpi=220)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--classifiers", nargs="+", default=["svm", "rf", "logreg"], choices=["svm", "rf", "logreg"])
    parser.add_argument("--text-reprs", nargs="+", default=["tfidf", "bert"], choices=["tfidf", "bert"])
    parser.add_argument("--pooling", default="mean", choices=["mean", "mean_std"])
    parser.add_argument("--tfidf-max-features", type=int, default=5000)
    parser.add_argument("--skip-stats", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(data_dir)
    folds = load_folds(data_dir, dataset.labels)
    print(f"Loaded {len(dataset.ids)} instances and {len(folds)} folds.")

    numeric_features: dict[str, np.ndarray] = {}
    audio = load_audio_features(data_dir, dataset.ids, pooling=args.pooling)
    if audio is not None:
        numeric_features["audio"] = audio
        print(f"Audio features: {audio.shape}")
    else:
        print("Audio features not found; skipping audio experiments.")

    vision = load_vision_features(data_dir, dataset.ids, pooling=args.pooling)
    if vision is not None:
        numeric_features["vision"] = vision
        print(f"Vision features: {vision.shape}")
    else:
        print("Vision HDF5 not found; skipping tri-modal experiments.")

    bert = load_bert_features(data_dir, expected_rows=len(dataset.ids))
    if bert is not None:
        numeric_features["bert"] = bert
        print(f"BERT features: {bert.shape}")
    elif "bert" in args.text_reprs:
        print("BERT features not found; BERT experiments will be skipped.")

    save_label_plot(dataset, output_dir)
    if not args.skip_stats:
        if "audio" in numeric_features:
            save_feature_tests("audio", numeric_features["audio"], dataset.labels, output_dir)
        if "vision" in numeric_features:
            save_feature_tests("vision", numeric_features["vision"], dataset.labels, output_dir)

    modalities = ["text"]
    if "audio" in numeric_features:
        modalities.append("text+audio")
    if "audio" in numeric_features and "vision" in numeric_features:
        modalities.append("text+audio+vision")

    fold_rows = []
    cms: dict[str, np.ndarray] = {}
    for text_repr in args.text_reprs:
        if text_repr == "bert" and "bert" not in numeric_features:
            continue
        for classifier in args.classifiers:
            for modality in modalities:
                print(f"Running {text_repr} | {classifier} | {modality}")
                rows, cm = run_experiment(
                    dataset=dataset,
                    folds=folds,
                    text_repr=text_repr,
                    modality=modality,
                    classifier_name=classifier,
                    data_dir=data_dir,
                    numeric_features=numeric_features,
                    output_dir=output_dir,
                    tfidf_max_features=args.tfidf_max_features,
                )
                fold_rows.extend(rows)
                key = text_repr.upper() + " + " + rows[0]["modality"] + " + " + classifier.upper()
                cms[key] = cm

    if not fold_rows:
        raise RuntimeError("No experiments ran. Check that text or feature files exist.")

    fold_df = pd.DataFrame(fold_rows)
    agg_df = aggregate_results(fold_df)
    fold_df.to_csv(output_dir / "fold_results.csv", index=False)
    agg_df.to_csv(output_dir / "summary_results.csv", index=False)
    save_result_plots(agg_df, fold_df, cms, output_dir)
    save_paper_comparison(agg_df, output_dir)

    print("\nSummary:")
    cols = ["model", "text_repr", "modality", "accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    print(agg_df[cols].sort_values("f1_weighted", ascending=False).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nSaved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
