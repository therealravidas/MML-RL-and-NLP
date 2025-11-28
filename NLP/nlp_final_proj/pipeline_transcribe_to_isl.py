#!/usr/bin/env python3
"""
pipeline_transcribe_to_isl.py

Integrates AssemblyAI transcription -> ISL assembler.

Usage (single audio file -> single ISL video):
  python pipeline_transcribe_to_isl.py \
    --audio /path/to/audio.wav \
    --clips_dir /path/to/isl_clips \
    --out ./out/isl_from_audio.mp4 \
    --workdir ./tmp_pipeline_work \
    --api_key YOUR_ASSEMBLYAI_KEY

Usage (use local assemblyai_transcript.py settings):
  python pipeline_transcribe_to_isl.py --audio /path/to/audio.wav --clips_dir ./isl_clips --out ./out.mp4

Notes:
 - This script attempts to import functions from two local files:
     assemblyai_transcript.py  (expects upload_file, start_transcription, wait_for_completion)
     assemble_isl_ffmpeg.py    (expects assemble_text_to_video)
   If they live at different filenames, pass --assemblyai / --assembler paths.
 - See inline logging; run from a terminal (not a notebook).
"""
import argparse
import importlib.util
import runpy
import sys
from pathlib import Path
import json
import time

# -----------------------
# Helpers to load modules by path
# -----------------------
def load_module_from_path(module_name, path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{module_name} not found at {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# -----------------------
# CLI
# -----------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True, help="local audio file to upload/transcribe")
    p.add_argument("--clips_dir", required=True, help="isl clips directory for assembler")
    p.add_argument("--out", required=True, help="output isl mp4 path")
    p.add_argument("--workdir", default=None, help="temporary workdir prefix (created if missing)")
    p.add_argument("--assemblyai", default="assemblyai_transcript.py", help="path to assemblyai_transcript.py")
    p.add_argument("--assembler", default="text2isl.py", help="path to ISL assembler module (assemble_text_to_video)")
    p.add_argument("--api_key", default=None, help="AssemblyAI API key (overrides file if assemblyai module has none)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--pause_clip", default="pause.mp4")
    p.add_argument("--inter_clip_pause", type=float, default=0.12)
    p.add_argument("--no_spell", action="store_true")
    args = p.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print("Audio file not found:", audio_path); sys.exit(1)

    # --- load assemblyai module ---
    print("[1/3] Loading AssemblyAI helper from", args.assemblyai)
    try:
        assembly_mod = load_module_from_path("assemblyai_transcript_module", args.assemblyai)
    except Exception as e:
        print("Failed to load assemblyai module:", e)
        raise

    # Validate expected functions
    if not all(hasattr(assembly_mod, name) for name in ("upload_file","start_transcription","wait_for_completion")):
        print("The assemblyai module must define: upload_file, start_transcription, wait_for_completion")
        print("Module functions found:", [n for n in dir(assembly_mod) if not n.startswith('_')][:50])
        raise SystemExit(1)

    # Optionally override API_KEY in the module (if they used a constant)
    if args.api_key:
        # set attribute API_KEY if exists else set a variable in module
        setattr(assembly_mod, "API_KEY", args.api_key)
        print("[info] Overrode assemblyai API_KEY in module with CLI value")

    # --- run transcription ---
    print("[2/3] Uploading audio and requesting transcription...")
    try:
        upload_url = assembly_mod.upload_file(str(audio_path))
        print(" upload_url:", upload_url)
    except Exception as e:
        print("upload_file raised:", e); raise

    try:
        transcript_id = assembly_mod.start_transcription(upload_url)
        print(" transcript_id:", transcript_id)
    except Exception as e:
        print("start_transcription raised:", e); raise

    print(" Waiting for completion (polling)...")
    text = assembly_mod.wait_for_completion(transcript_id)
    print("\n--- Transcript received ---\n")
    print(text)
    print("\n--------------------------\n")

    # --- load assembler module ---
    print("[3/3] Loading ISL assembler from", args.assembler)
    try:
        assembler_mod = load_module_from_path("assemble_isl_module", args.assembler)
    except Exception as e:
        print("Failed to load assembler module:", e)
        raise

    # Check assembler API
    if not hasattr(assembler_mod, "assemble_text_to_video"):
        # older script may expose assemble_text_to_video under another name; try to find
        candidates = [n for n in dir(assembler_mod) if "assemble" in n.lower()]
        print("assemble_text_to_video not found. Candidates:", candidates)
        raise SystemExit("Assembler module must provide assemble_text_to_video(text, clips_dir, out_mp4, workdir, ...).")

    # prepare workdir
    if args.workdir:
        workdir = args.workdir
    else:
        # safe timestamped workdir within current dir
        workdir = f"./tmp_pipeline_work_{int(time.time())}"
    Path(workdir).mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Calling assembler...")
    try:
        segs, used_workdir = assembler_mod.assemble_text_to_video(
            text,
            args.clips_dir,
            str(out_path),
            workdir,
            width=args.width,
            height=args.height,
            fps=args.fps,
            pause_clip_name=args.pause_clip,
            inter_pause=args.inter_clip_pause,
            fallback_spell=not args.no_spell,
            font_path=getattr(args, "font_path", None)  # optional
        )
    except TypeError:
        # some versions return only segs (not segs,wd) — try fallback signature
        try:
            res = assembler_mod.assemble_text_to_video(
                text,
                args.clips_dir,
                str(out_path),
                workdir,
                width=args.width,
                height=args.height,
                fps=args.fps,
                pause_clip_name=args.pause_clip,
                inter_pause=args.inter_clip_pause,
                fallback_spell=not args.no_spell,
                font_path=getattr(args, "font_path", None)
            )
            # if res is list -> segments; we can't get workdir
            segs = res if isinstance(res, list) else res[0]
            used_workdir = workdir
        except Exception as e:
            print("Failed to call assembler with fallback signature:", e)
            raise
    except Exception as e:
        print("Assembler raised exception:", e)
        raise

    print("Assembled ISL video at:", out_path)
    print("Workdir:", used_workdir)
    # write manifest
    manifest = {"audio": str(audio_path), "transcript_id": transcript_id, "transcript": text, "isl_video": str(out_path), "segments": segs, "workdir": used_workdir}
    mpath = out_path.with_suffix(".manifest.json")
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("Wrote manifest:", mpath)

if __name__ == "__main__":
    main()
