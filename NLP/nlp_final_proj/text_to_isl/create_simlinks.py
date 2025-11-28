#!/usr/bin/env python3
# create_normalized_symlinks.py
import re
from pathlib import Path
import argparse

def token_to_filename(token: str) -> str:
    t = token.strip().lower()
    t = re.sub(r"[^a-z0-9]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t

p = argparse.ArgumentParser()
p.add_argument("--clips_dir", required=True, help="path to isl_clips folder")
p.add_argument("--dry", action="store_true", help="just print actions without creating symlinks")
args = p.parse_args()

clips = Path(args.clips_dir)
if not clips.exists():
    raise SystemExit("clips_dir not found: " + str(clips))

mp4s = [f for f in clips.iterdir() if f.is_file() and f.suffix.lower()=='.mp4']
print(f"Found {len(mp4s)} mp4 files in {clips}")

for f in mp4s:
    stem = f.stem  # original name without .mp4
    normalized = token_to_filename(stem) + ".mp4"
    linkp = clips / normalized
    if linkp.exists():
        # if exists and is same file (hardlink/symlink), skip
        try:
            if linkp.resolve() == f.resolve():
                print("OK (exists same):", linkp.name)
                continue
        except Exception:
            pass
        # if exists but different, skip to avoid overwriting
        print("SKIP (normalized exists and different):", linkp.name, "->", f.name)
        continue
    print(("DRY: would link" if args.dry else "LINK:"), normalized, "<--", f.name)
    if not args.dry:
        try:
            # create a relative symlink to keep folder portable
            linkp.symlink_to(f.name)
        except Exception as e:
            print("Failed to symlink", f, ":", e)
