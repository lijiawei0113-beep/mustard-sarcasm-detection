#!/usr/bin/env python
"""Download optional MUStARD pre-extracted BERT and visual features.

The core experiment can run with the small GitHub files already in data/.
Use this script only when you want BERT and/or tri-modal visual runs.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


HF_BASE = "https://huggingface.co/datasets/MichiganNLP/MUStARD/resolve/main"
FILES = {
    "bert": ("BERT_text_features.zip", Path("BERT_text_features.zip")),
    "vision_utterances": (
        "features/utterances_final/resnet_pool5.hdf5",
        Path("features/utterances_final/resnet_pool5.hdf5"),
    ),
    "vision_context": (
        "features/context_final/resnet_pool5.hdf5",
        Path("features/context_final/resnet_pool5.hdf5"),
    ),
}


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already exists: {destination}")
        return
    tmp = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as file:
        shutil.copyfileobj(response, file)
    tmp.replace(destination)
    print(f"Saved {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--bert", action="store_true", help="Download and unzip BERT_text_features.zip.")
    parser.add_argument("--vision", action="store_true", help="Download utterance and context ResNet HDF5 files.")
    args = parser.parse_args()

    if not args.bert and not args.vision:
        parser.error("Choose --bert, --vision, or both.")

    if args.bert:
        filename, relative = FILES["bert"]
        archive = args.data_dir / relative
        download(f"{HF_BASE}/{filename}?download=true", archive)
        with zipfile.ZipFile(archive) as zip_file:
            names = zip_file.namelist()
            print("Unzipping BERT features:", ", ".join(names[:5]))
            zip_file.extractall(args.data_dir)
        print("BERT features should now include data/bert-output.jsonl.")

    if args.vision:
        for key in ["vision_utterances", "vision_context"]:
            hf_path, relative = FILES[key]
            download(f"{HF_BASE}/{hf_path}?download=true", args.data_dir / relative)
        print("Vision features should now include data/features/utterances_final/resnet_pool5.hdf5.")


if __name__ == "__main__":
    main()
