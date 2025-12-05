#!/usr/bin/env python3
"""
Resolve audio paths in a JSONL and write a new JSONL with absolute 'audio_path' fields where possible.

Usage:
python fix_jsonl_make_absolute.py \
  --json /path/to/train.jsonl \
  --out /path/to/train.resolved.jsonl \
  --candidates /home/.../dirty_sa /home/.../pspeech_wavs
"""
import argparse, json, os
from pathlib import Path

def try_resolve(candidate_paths, ap):
    if not ap:
        return None
    p = Path(ap)
    if p.is_absolute() and p.exists():
        return str(p.resolve())
    for root in candidate_paths:
        cand = (Path(root) / ap).resolve()
        if cand.exists():
            return str(cand)
    if Path(ap).exists():
        return str(Path(ap).resolve())
    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--candidates", nargs="*", default=[])
    args = p.parse_args()

    inp = Path(args.json)
    outp = Path(args.out)
    candidates = [Path(x).expanduser().resolve() for x in args.candidates]

    with inp.open() as fin, outp.open("w", encoding="utf8") as fout:
        for i, line in enumerate(fin):
            try:
                obj = json.loads(line)
            except:
                continue
            audio_candidate = None
            for k in ("audio_path","audio","path","file","file_path","audio_filepath"):
                if k in obj and obj[k] is not None:
                    audio_candidate = obj[k]
                    audio_key = k
                    break
            if audio_candidate is None:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            # if dict with 'path'
            resolved = None
            if isinstance(audio_candidate, dict) and "path" in audio_candidate:
                resolved = try_resolve(candidates, audio_candidate["path"])
            elif isinstance(audio_candidate, str):
                resolved = try_resolve(candidates, audio_candidate)

            if resolved:
                # write/update audio_path key for consistency
                obj["audio_path"] = resolved
            # otherwise leave object as-is (could be array)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("Wrote resolved JSONL to", outp)

if __name__ == "__main__":
    main()
