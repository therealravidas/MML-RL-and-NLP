#!/usr/bin/env python3
"""
Stream-processing of parquet shards into JSONL metadata files (and optional wav extraction).

Usage example:
python stream_metadata_from_parquets.py \
  --base "/home/BTECH_7TH_SEM/Desktop/peoples_speech_dirty_sa/dirty_sa" \
  --out_meta "/home/BTECH_7TH_SEM/pspeech_meta_jsonl" \
  --out_audio "/home/BTECH_7TH_SEM/pspeech_wavs" \
  --batch_size 128 \
  --extract_arrays

Notes:
- Run from terminal (not inside VS Code UI or Jupyter) to avoid UI/kernel crashes.
- If your parquet rows already contain file paths to audio, --extract_arrays is not needed.
"""
import argparse
from pathlib import Path
import json
import uuid
import sys
import os
import math
import soundfile as sf
import numpy as np
import pyarrow.parquet as pq

def infer_split_from_name(name: str):
    n = name.lower()
    if "train" in n:
        return "train"
    if "validation" in n or "valid" in n:
        return "validation"
    if "test" in n:
        return "test"
    return None

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def write_jsonl_line(fp, obj):
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")

def try_resolve_path(base: Path, candidate: str):
    p = Path(candidate)
    if p.is_absolute():
        return str(p) if p.exists() else None
    # try relative to base
    r = (base / candidate).resolve()
    if r.exists():
        return str(r)
    # try candidate as-is without resolving
    if (Path(candidate)).exists():
        return str(Path(candidate).resolve())
    return None

def process_parquet_file(pq_path: Path, base: Path, out_audio_dir: Path, jsonl_files: dict,
                         extract_arrays: bool, batch_size: int, stats: dict):
    pf = pq.ParquetFile(str(pq_path))
    n_row_groups = pf.metadata.num_row_groups
    # iterate row groups using iter_batches (pyarrow)
    it = pf.iter_batches(batch_size=batch_size)
    batch_index = 0
    for batch in it:
        batch_index += 1
        # convert to python-friendly dict of columns -> lists
        try:
            batch_dict = batch.to_pydict()
        except Exception as e:
            print(f"[WARN] failed to to_pydict() on {pq_path} batch {batch_index}: {e}")
            continue
        n_rows = 0
        # the batch_dict has lists of column values; iterate rows by index
        columns = list(batch_dict.keys())
        if not columns:
            continue
        n_rows = len(batch_dict[columns[0]])
        for i in range(n_rows):
            row = {k: batch_dict[k][i] for k in columns}
            # find transcript
            transcript = None
            for tkey in ("transcript", "text", "sentence", "utterance", "caption"):
                v = row.get(tkey)
                if v is not None and isinstance(v, str) and len(v.strip())>0:
                    transcript = v
                    break
            if transcript is None:
                transcript = ""  # allow empty, but user may filter later

            # find audio path candidate or array
            audio_path = None
            audio_array = None
            sr = None

            # common audio keys
            for akey in ("audio", "audio_filepath", "audio_path", "path", "file", "file_path"):
                if akey in row and row[akey] is not None:
                    val = row[akey]
                    # dict with 'path' or 'array'
                    if isinstance(val, dict):
                        if "path" in val and isinstance(val["path"], str):
                            audio_path = val["path"]
                        elif "array" in val:
                            audio_array = val.get("array")
                            sr = val.get("sampling_rate") or val.get("sampling-rate") or val.get("samplingRate") or None
                    elif isinstance(val, str) and val.lower().endswith((".wav", ".flac", ".mp3")):
                        audio_path = val
                    # else could be bytes/array in other representations; we'll detect below
                if audio_path or audio_array is not None:
                    break

            # fallback: scan all fields for string that looks like audio filename
            if audio_path is None and audio_array is None:
                for k, v in row.items():
                    if isinstance(v, str) and v.lower().endswith((".wav", ".flac", ".mp3")):
                        audio_path = v
                        break
                    # sometimes arrays come as lists of floats
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (float, int)):
                        # naive detection: treat as array only if sampling rate field exists somewhere
                        audio_array = v
                        # sr is unknown here; we'll default to 16000 later if extracting
                        break

            # resolve audio_path to absolute existing path if possible
            resolved_path = None
            if audio_path:
                resolved_path = try_resolve_path(base, audio_path)
            if resolved_path is None and audio_path:
                # keep the raw audio_path string even if file not found; user may adjust manually later
                resolved_path = audio_path

            out_obj = None

            if resolved_path:
                out_obj = {"audio_path": resolved_path, "transcript": transcript}
                stats["rows_with_path"] += 1
            elif audio_array is not None and extract_arrays:
                # write array to wav (streamed, one-by-one)
                try:
                    # audio_array may be list or numpy array
                    arr = np.asarray(audio_array, dtype="float32")
                    if arr.ndim > 1:
                        # flatten or average channels
                        arr = arr.mean(axis=0)
                    # determine sr
                    local_sr = int(sr) if sr else 16000
                    fname = f"{uuid.uuid4().hex}.wav"
                    out_path = out_audio_dir / fname
                    sf.write(str(out_path), arr, local_sr)
                    out_obj = {"audio_path": str(out_path), "transcript": transcript}
                    stats["rows_extracted_arrays"] += 1
                except Exception as e:
                    stats["rows_audio_fail"] += 1
                    # skip writing audio for this row
                    continue
            else:
                stats["rows_skipped"] += 1
                continue

            # determine which split this parquet belongs to (infer from filename)
            split = infer_split_from_name(pq_path.name) or "train"
            # append line to correct jsonl file
            write_jsonl_line(jsonl_files[split], out_obj)
            stats[f"{split}_count"] += 1

        # flush after each batch to keep file system updated and low memory
        for f in jsonl_files.values():
            f.flush()
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Folder containing parquet shards")
    parser.add_argument("--out_meta", required=True, help="Output folder for JSONL metadata (one JSONL per split)")
    parser.add_argument("--out_audio", required=False, default=None, help="Optional folder to write extracted wavs (if parquet stores arrays).")
    parser.add_argument("--batch_size", type=int, default=256, help="Rows per batch (lower to reduce memory/IO).")
    parser.add_argument("--extract_arrays", action="store_true", help="If present, extract array blobs to wav files.")
    args = parser.parse_args()

    BASE = Path(args.base).expanduser().resolve()
    OUT_META = Path(args.out_meta).expanduser().resolve()
    OUT_AUDIO = Path(args.out_audio).expanduser().resolve() if args.out_audio else None
    ensure_dir(OUT_META)
    if OUT_AUDIO and args.extract_arrays:
        ensure_dir(OUT_AUDIO)

    # open jsonl files (append mode) for each split
    jsonl_paths = {
        "train": OUT_META / "train.jsonl",
        "validation": OUT_META / "validation.jsonl",
        "test": OUT_META / "test.jsonl",
    }
    jsonl_files = {}
    for k, p in jsonl_paths.items():
        jsonl_files[k] = open(p, "a", encoding="utf8")  # append so reruns won't clobber by default

    parquets = sorted([p for p in BASE.rglob("*.parquet")])
    if not parquets:
        print("No parquet files found under", BASE)
        for f in jsonl_files.values():
            f.close()
        sys.exit(1)

    stats = {
        "total_shards": len(parquets),
        "rows_with_path": 0,
        "rows_extracted_arrays": 0,
        "rows_skipped": 0,
        "rows_audio_fail": 0,
        "train_count": 0,
        "validation_count": 0,
        "test_count": 0,
    }

    print(f"Found {len(parquets)} parquet files. Processing with batch_size={args.batch_size}")
    for idx, pqf in enumerate(parquets, 1):
        try:
            print(f"[{idx}/{len(parquets)}] Processing: {pqf.name}")
            process_parquet_file(pqf, BASE, OUT_AUDIO, jsonl_files, args.extract_arrays, args.batch_size, stats)
        except Exception as e:
            print(f"[ERROR] Failed processing {pqf}: {e}")

    for f in jsonl_files.values():
        f.close()

    print("Done. Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("\nCreated JSONL metadata files (one per split) at:", OUT_META)
    if OUT_AUDIO:
        print("Extracted wavs (if any) at:", OUT_AUDIO)
    print("To load metadata into HF datasets, use:\nfrom datasets import load_dataset\nload_dataset('json', data_files={'train':'/path/to/train.jsonl', 'validation':'...','test':'...'})")

if __name__ == '__main__':
    main()
