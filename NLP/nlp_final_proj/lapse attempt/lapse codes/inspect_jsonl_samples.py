#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path

def short(x, n=200):
    s = repr(x)
    return s if len(s) <= n else s[:n] + "...[truncated]..."

def inspect_obj(obj):
    print("  Keys:", list(obj.keys())[:30])
    # candidate audio-like keys to inspect
    keys = ["audio", "audio_path", "audio_filepath", "path", "file", "waveform", "samples", "array", "bytes", "blob"]
    for k in keys:
        if k in obj:
            print(f"  -> {k}: type={type(obj[k]).__name__}")
            print("     value:", short(obj[k], n=400))
    # also print types of all top-level fields
    print("  Field types (first 20):")
    for i, (k,v) in enumerate(obj.items()):
        if i>=20: break
        print(f"    {k} : {type(v).__name__} -> {short(v, n=150)}")
    print("-"*80)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True, help="JSONL file to inspect")
    p.add_argument("--n", type=int, default=10, help="How many lines to inspect")
    args = p.parse_args()

    js = Path(args.json)
    if not js.exists():
        print("File not found:", js); sys.exit(1)

    with js.open("r", encoding="utf8") as f:
        for i, line in enumerate(f, 1):
            if i > args.n:
                break
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"Line {i}: failed to parse JSON: {e}")
                continue
            print(f"--- LINE {i} ---")
            inspect_obj(obj)

if __name__ == "__main__":
    main()
