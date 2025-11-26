#!/usr/bin/env python3
"""
Build metadata HF datasets from parquet shards that are named like:
  train-00000-of-00008.parquet
  test-00000-of-00008.parquet
  validation-00000-of-00008.parquet

Usage:
  python build_metadata_from_parquets.py --base "/home/BTECH_7TH_SEM/Desktop/peoples_speech_dirty_sa/dirty_sa" --out "/home/BTECH_7TH_SEM/pspeech_meta"
"""
import argparse
from pathlib import Path
from datasets import load_dataset, Dataset
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--base", required=True, help="Folder containing parquet shards")
parser.add_argument("--out", required=True, help="Output folder to save metadata datasets")
args = parser.parse_args()

BASE = Path(args.base).expanduser().resolve()
OUT = Path(args.out).expanduser().resolve()
OUT.mkdir(parents=True, exist_ok=True)

if not BASE.exists():
    print("Base folder not found:", BASE)
    sys.exit(1)

# helper to infer split from filename
def infer_split_from_name(name):
    n = name.lower()
    if "train" in n:
        return "train"
    if "validation" in n or "valid" in n:
        return "validation"
    if "test" in n:
        return "test"
    # fallback
    return None

# collect parquet files and group by split
parquets = sorted([p for p in BASE.rglob("*.parquet")])
if not parquets:
    print("No parquet files found under", BASE)
    sys.exit(1)

groups = {"train": [], "validation": [], "test": [], "unknown": []}
for p in parquets:
    split = infer_split_from_name(p.name)
    if split is None:
        groups["unknown"].append(str(p))
    else:
        groups[split].append(str(p))

# move unknowns to 'train' as fallback (or leave them separate)
if groups["unknown"]:
    print(f"Warning: {len(groups['unknown'])} parquet(s) with unknown split — assigning to 'train' by default.")
    groups["train"].extend(groups["unknown"])
    groups.pop("unknown", None)

print("Shard counts:", {k: len(v) for k, v in groups.items()})

# function to process a list of parquet files into metadata rows
def parquet_to_metadata(file_list, split_name):
    if not file_list:
        print(f"No shards for split {split_name}, skipping.")
        return None
    print(f"Loading {len(file_list)} shard(s) for split {split_name} ...")
    # load as HF dataset (parquet loader)
    ds = load_dataset("parquet", data_files=file_list, split="train")  # returns Dataset
    print(f"Loaded {len(ds)} rows; columns: {ds.column_names}")
    rows = []
    for i, ex in enumerate(ds):
        # Attempt to extract audio path and transcript using common patterns
        audio_candidate = None
        transcript = ""
        # common column names to check
        possible_audio_keys = ["audio", "audio_filepath", "audio_path", "path", "file", "file_path"]
        possible_text_keys = ["transcript", "text", "sentence", "utterance", "caption"]

        # pick transcript if present
        for k in possible_text_keys:
            if k in ex and ex[k] is not None:
                transcript = ex[k]
                break

        # find audio path: many datasets store audio as dict with 'path' or as string path
        for k in possible_audio_keys:
            if k in ex and ex[k] is not None:
                val = ex[k]
                # dict with 'path'
                if isinstance(val, dict) and "path" in val and isinstance(val["path"], str):
                    audio_candidate = val["path"]
                    break
                # string path
                if isinstance(val, str) and val.lower().endswith((".wav", ".flac", ".mp3")):
                    audio_candidate = val
                    break
                # sometimes audio stored as {"array": [...], "sampling_rate": 16000}
                # we skip array blobs here to avoid memory decoding; handle separately if needed
        # fallback: scan all fields for a string filename
        if audio_candidate is None:
            for k, v in ex.items():
                if isinstance(v, str) and v.lower().endswith((".wav", ".flac", ".mp3")):
                    audio_candidate = v
                    break

        if audio_candidate is None:
            # skip rows that don't expose a file path (to avoid decoding arrays here)
            continue

        # make absolute if necessary - try several bases:
        ap = Path(audio_candidate)
        if not ap.is_absolute():
            # try relative to BASE (where parquet sits)
            # sometimes parquet stores only filenames relative to a common root
            candidate_abs = (BASE / audio_candidate).resolve()
            if candidate_abs.exists():
                audio_candidate = str(candidate_abs)
            else:
                # try relative to the parquet file's parent folder (safer)
                # find which parquet shard contained this example and use its parent
                # we don't have the specific shard path in this loop easily; assume BASE
                audio_candidate = str(candidate_abs)  # keep this even if it doesn't exist; user can adjust
        rows.append({"audio_path": audio_candidate, "transcript": transcript})

    if not rows:
        print(f"Warning: no usable rows with file paths found for split {split_name}.")
        return None
    meta_ds = Dataset.from_list(rows)
    return meta_ds

# Build and save metadata datasets
for split in ["train", "validation", "test"]:
    meta = parquet_to_metadata(groups.get(split, []), split)
    if meta is None:
        continue
    out_path = OUT / split
    meta.save_to_disk(str(out_path))
    print(f"Saved metadata for {split}: {len(meta)} examples -> {out_path}")

print("Done.")
