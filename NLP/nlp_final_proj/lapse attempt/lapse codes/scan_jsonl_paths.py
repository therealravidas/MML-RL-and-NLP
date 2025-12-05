#!/usr/bin/env python3
"""
Scan a JSONL metadata file and report:
 - how many rows have an existing audio path
 - how many rows contain arrays (lists of floats)
 - how many rows are unresolved
It also attempts to resolve relative paths against candidate base directories.

Usage:
python scan_jsonl_paths.py --json /path/to/train.jsonl --candidates /path/to/snapshot /other/candidate
"""
import argparse, json, os
from pathlib import Path
from collections import Counter

def try_resolve(candidate_paths, ap):
    # try absolute first
    if not ap:
        return None
    p = Path(ap)
    if p.is_absolute() and p.exists():
        return str(p.resolve())
    # otherwise try candidate roots
    for root in candidate_paths:
        cand = (Path(root) / ap).resolve()
        if cand.exists():
            return str(cand)
    # try the path as-is (maybe relative to cwd)
    if Path(ap).exists():
        return str(Path(ap).resolve())
    return None

def looks_like_array(x):
    # naive check for a list-like of numbers
    if isinstance(x, list) and len(x)>0 and isinstance(x[0], (int, float)):
        return True
    if isinstance(x, dict) and ("array" in x) and isinstance(x["array"], list):
        return True
    return False

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True)
    p.add_argument("--candidates", nargs="*", default=[], help="candidate root folders to resolve relative audio paths")
    p.add_argument("--sample", type=int, default=10, help="how many unresolved samples to print")
    args = p.parse_args()

    js = Path(args.json)
    if not js.exists():
        print("JSONL not found:", js); return

    candidate_paths = [Path(x).expanduser().resolve() for x in args.candidates]

    totals = Counter()
    unresolved_examples = []
    with js.open() as f:
        for i, line in enumerate(f):
            try:
                obj = json.loads(line)
            except Exception as e:
                totals["bad_json"] += 1
                continue
            # find audio-like keys
            audio_candidate = None
            for k in ("audio_path","audio","path","file","file_path","audio_filepath","audio_filepath"):
                if k in obj and obj[k] is not None:
                    audio_candidate = obj[k]
                    break
            if audio_candidate is None:
                totals["no_audio_key"] += 1
                if len(unresolved_examples) < args.sample:
                    unresolved_examples.append((i, obj))
                continue

            # if dict with 'path'
            if isinstance(audio_candidate, dict) and "path" in audio_candidate:
                path_val = audio_candidate["path"]
                resolved = try_resolve(candidate_paths, path_val)
                if resolved:
                    totals["existing_path"] += 1
                else:
                    # maybe array inside dict
                    if looks_like_array(audio_candidate):
                        totals["has_array"] += 1
                    else:
                        totals["unresolved_path"] += 1
                        if len(unresolved_examples) < args.sample:
                            unresolved_examples.append((i, {"audio_candidate": audio_candidate}))
            elif isinstance(audio_candidate, str):
                resolved = try_resolve(candidate_paths, audio_candidate)
                if resolved:
                    totals["existing_path"] += 1
                else:
                    totals["unresolved_path"] += 1
                    if len(unresolved_examples) < args.sample:
                        unresolved_examples.append((i, {"audio_candidate": audio_candidate}))
            elif looks_like_array(audio_candidate):
                totals["has_array"] += 1
            else:
                totals["unknown_audio_format"] += 1
                if len(unresolved_examples) < args.sample:
                    unresolved_examples.append((i, {"audio_candidate": audio_candidate}))

    print("Scan report for", js)
    for k,v in totals.items():
        print(f"  {k}: {v}")
    print("\nSample unresolved examples (index, excerpt):")
    for idx, ex in unresolved_examples:
        print("INDEX", idx, ex)
    print("\nCandidate roots tried:", [str(p) for p in candidate_paths])

if __name__ == "__main__":
    main()
