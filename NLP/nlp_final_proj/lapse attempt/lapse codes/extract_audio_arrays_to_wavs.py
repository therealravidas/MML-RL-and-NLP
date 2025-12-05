#!/usr/bin/env python3
"""
Stream-extract embedded audio arrays from a JSONL metadata file to WAV files,
and produce a new JSONL where each entry has an 'audio_path' if possible.

Usage:
python extract_audio_arrays_to_wavs.py \
  --input /home/BTECH_7TH_SEM/pspeech_meta_jsonl/train.resolved.jsonl \
  --output /home/BTECH_7TH_SEM/pspeech_meta_jsonl/train.extracted.jsonl \
  --out_wavs /home/BTECH_7TH_SEM/pspeech_wavs \
  --batch_size 64

Notes:
- Run in a terminal (not VS Code UI) to avoid editor crashes.
- Default sampling rate if none present: 16000 Hz.
- This script never loads many arrays at once; it processes rows one-by-one.
"""
import argparse, json, os, uuid, sys
from pathlib import Path
import soundfile as sf
import numpy as np

def is_array_like(x):
    # Detect inline arrays or dicts containing an 'array' field
    if isinstance(x, list) and len(x) > 0 and isinstance(x[0], (int, float)):
        return True
    if isinstance(x, dict) and "array" in x:
        return True
    return False

def get_array_and_sr(field):
    # field might be: list of floats, or dict {'array': [...], 'sampling_rate': 16000}
    if isinstance(field, dict):
        arr = field.get("array") or field.get("data") or None
        sr = field.get("sampling_rate") or field.get("sampling-rate") or field.get("sr") or None
    elif isinstance(field, list):
        arr = field
        sr = None
    else:
        return None, None
    if arr is None:
        return None, None
    # numpy conversion
    arr_np = np.asarray(arr, dtype="float32")
    # If 2D, reduce to mono by averaging channels
    if arr_np.ndim > 1:
        arr_np = arr_np.mean(axis=0)
    return arr_np, int(sr) if sr else None

def try_get_path_from_obj(obj, candidate_keys):
    # return existing absolute path if present and exists
    for k in candidate_keys:
        if k in obj and obj[k]:
            v = obj[k]
            if isinstance(v, dict) and "path" in v and isinstance(v["path"], str):
                p = Path(v["path"])
            elif isinstance(v, str):
                p = Path(v)
            else:
                continue
            if p.exists():
                return str(p.resolve())
    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input JSONL (may contain arrays or paths)")
    p.add_argument("--output", required=True, help="Output JSONL with audio_path fields")
    p.add_argument("--out_wavs", required=True, help="Directory to write extracted wavs")
    p.add_argument("--batch_size", type=int, default=64, help="How many lines to process before flushing logs (keeps memory low)")
    p.add_argument("--default_sr", type=int, default=16000, help="Default sample rate for arrays without SR")
    args = p.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)
    out_wavs = Path(args.out_wavs)
    out_wavs.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        print("Input JSONL not found:", inp); sys.exit(1)

    candidate_audio_keys = ["audio_path","audio","path","file","file_path","audio_filepath","audio_filepath"]
    processed = 0
    rows_with_path = 0
    rows_extracted = 0
    rows_skipped = 0

    with inp.open("r", encoding="utf8") as fin, outp.open("w", encoding="utf8") as fout:
        for i, line in enumerate(fin, 1):
            try:
                obj = json.loads(line)
            except Exception as e:
                # skip bad json line
                rows_skipped += 1
                continue

            # Prefer an already-existing path
            existing = try_get_path_from_obj(obj, candidate_audio_keys)
            if existing:
                obj["audio_path"] = existing
                rows_with_path += 1
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            else:
                # find audio field that is array-like
                audio_field = None
                audio_key_found = None
                for k in candidate_audio_keys:
                    if k in obj and obj[k] is not None:
                        if is_array_like(obj[k]):
                            audio_field = obj[k]
                            audio_key_found = k
                            break
                        # sometimes obj[k] is dict with 'array'
                        if isinstance(obj[k], dict) and "array" in obj[k]:
                            audio_field = obj[k]
                            audio_key_found = k
                            break
                if audio_field is not None:
                    arr, sr = get_array_and_sr(audio_field)
                    if arr is None:
                        rows_skipped += 1
                        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        continue
                    if sr is None:
                        sr = args.default_sr
                    # create a unique wav name and write to disk
                    fname = f"{uuid.uuid4().hex}.wav"
                    out_file = out_wavs / fname
                    try:
                        sf.write(str(out_file), arr, sr)
                        obj["audio_path"] = str(out_file.resolve())
                        # optional: remove large array from object to save space in output JSONL
                        if audio_key_found in obj:
                            # keep original under audio_array (optional)
                            try:
                                del obj[audio_key_found]
                            except Exception:
                                pass
                        rows_extracted += 1
                        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    except Exception as e:
                        print(f"[WARN] Failed to write wav for row {i}: {e}")
                        rows_skipped += 1
                        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                else:
                    # no path and no array found: skip or write as-is
                    rows_skipped += 1
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

            processed += 1
            # flush and print progress every batch_size
            if processed % args.batch_size == 0:
                fout.flush()
                print(f"Processed {processed} rows — path:{rows_with_path} extracted:{rows_extracted} skipped:{rows_skipped}")

    print("Done. Total processed:", processed)
    print("rows_with_path:", rows_with_path)
    print("rows_extracted:", rows_extracted)
    print("rows_skipped:", rows_skipped)
    print("Output JSONL:", outp)
    print("Extracted wavs (if any) in:", out_wavs)

if __name__ == "__main__":
    main()
