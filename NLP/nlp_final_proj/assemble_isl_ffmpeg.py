#!/usr/bin/env python3
"""
assemble_isl_ffmpeg.py

PIL (Pillow >=10) + ffmpeg pipeline for assembling ISL videos from text.
NO ImageMagick, NO MoviePy.

Requirements:
  - ffmpeg and ffprobe available in PATH
  - pip install pillow tqdm
  - Python 3.8+

Usage (single):
  python assemble_isl_ffmpeg.py --clips_dir ./isl_clips --text "hello world" --out ./out.mp4

Usage (batch):
  python assemble_isl_ffmpeg.py --clips_dir ./isl_clips --batch_in transcripts.jsonl --out_dir ./out_videos

Notes:
 - The script creates a temporary workdir (default ./tmp_isl_work_TIMESTAMP) and leaves files for debugging.
 - Normalizes clips to specified width/height/fps for safe concatenation.
"""
import argparse
import re
import json
import os
import subprocess
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# -----------------------
# Utilities
# -----------------------
def run(cmd, check=True):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{proc.stderr}")
    return proc

def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def token_to_filename(token: str) -> str:
    t = token.strip().lower()
    t = re.sub(r"[^a-z0-9]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t

def text_to_tokens(text: str):
    return re.findall(r"[A-Za-z0-9]+", text)

# ---------- robust clip lookup ----------
from difflib import get_close_matches
from pathlib import Path

def find_clip_for_token(token: str, clips_dir: Path):
    """
    Robust matching order:
      1) normalized exact filename: token_to_filename(token) + '.mp4'
      2) case-insensitive stem exact (e.g. 'Please.mp4' matches 'please')
      3) filename-token equality (filename contains the token as a separate token)
      4) substring match (token appears inside filename)
      5) fuzzy match (difflib)
    Returns (Path_or_None, match_kind_str)
    """
    token_name = token_to_filename(token)
    # 1) normalized exact
    p = clips_dir / f"{token_name}.mp4"
    if p.exists():
        return p, "exact_normalized"
    # 2) case-insensitive stem exact
    for f in clips_dir.iterdir():
        if not f.is_file() or f.suffix.lower() != ".mp4":
            continue
        if f.stem.lower() == token.lower():
            return f, "exact_stem_ci"
    # 3) filename token equality (split filename on non-alnum)
    for f in clips_dir.iterdir():
        if not f.is_file() or f.suffix.lower() != ".mp4":
            continue
        name_tokens = re.findall(r"[A-Za-z0-9]+", f.stem.lower())
        if token.lower() in name_tokens:
            return f, "filename_token"
    # 4) substring
    for f in clips_dir.iterdir():
        if not f.is_file() or f.suffix.lower() != ".mp4":
            continue
        if token_name in f.name.lower() or token.lower() in f.name.lower():
            return f, "substring"
    # 5) fuzzy match
    basenames = [f.name for f in clips_dir.iterdir() if f.is_file() and f.suffix.lower()==".mp4"]
    matches = get_close_matches(token_name + ".mp4", basenames, n=1, cutoff=0.65)
    if matches:
        return clips_dir / matches[0], "fuzzy"
    return None, None

# -----------------------
# Pillow text helpers (Pillow>=10 compatible)
# -----------------------
def get_text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width, height

def make_text_image(text: str, size=(1280,720), fontsize=48, font_path=None, color="white", bg="black"):
    w, h = size
    img = Image.new("RGB", (w, h), color=bg)
    draw = ImageDraw.Draw(img)

    # load font (fallback to DejaVu or default)
    try:
        if font_path and Path(font_path).exists():
            font = ImageFont.truetype(str(font_path), fontsize)
        else:
            font = ImageFont.truetype("DejaVuSans.ttf", fontsize)
    except Exception:
        font = ImageFont.load_default()

    # simple word-wrapping
    words = text.split()
    lines = []
    if not words:
        lines = [""]
    else:
        cur = words[0]
        for w_ in words[1:]:
            test = cur + " " + w_
            tw, _ = get_text_size(draw, test, font)
            if tw <= (w - 60):
                cur = test
            else:
                lines.append(cur)
                cur = w_
        lines.append(cur)

    _, line_h = get_text_size(draw, "Ay", font)
    total_h = line_h * len(lines)
    y = max((h - total_h) // 2, 10)
    for line in lines:
        line_w, _ = get_text_size(draw, line, font)
        x = max((w - line_w) // 2, 10)
        draw.text((x, y), line, font=font, fill=color)
        y += line_h

    return img

# -----------------------
# ffmpeg helpers
# -----------------------
def image_to_mp4(img, out_mp4: Path, duration=1.0, fps=30, size=(1280,720)):
    tmp_png = out_mp4.with_suffix(".png")
    img = img.convert("RGB").resize(size)
    img.save(tmp_png)
    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", str(tmp_png),
        "-c:v", "libx264", "-t", str(duration), "-vf", f"scale={size[0]}:{size[1]}",
        "-r", str(fps), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_mp4)
    ]
    run(cmd)
    try:
        tmp_png.unlink()
    except Exception:
        pass
    return out_mp4

def normalize_clip_to_mp4(src_path: Path, out_mp4: Path, size=(1280,720), fps=30):
    cmd = [
        "ffmpeg", "-y", "-i", str(src_path),
        "-vf", f"scale={size[0]}:{size[1]}:force_original_aspect_ratio=decrease,pad={size[0]}:{size[1]}:(ow-iw)/2:(oh-ih)/2,setsar=1",
        "-r", str(fps),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_mp4)
    ]
    run(cmd)
    return out_mp4

def create_black_mp4(path: Path, duration=0.2, size=(1280,720), fps=30):
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=size={size[0]}x{size[1]}:color=black:duration={duration}",
        "-c:v", "libx264", "-r", str(fps), "-pix_fmt", "yuv420p", str(path)
    ]
    run(cmd)
    return path

def get_video_duration(path: Path):
    # try stream duration then format duration
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = proc.stdout.strip()
    try:
        return float(out)
    except Exception:
        cmd2 = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
        proc2 = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out2 = proc2.stdout.strip()
        try:
            return float(out2)
        except Exception:
            return 0.0

# -----------------------
# Assembler core (single text)
# -----------------------
def assemble_text_to_video(text: str, clips_dir: str, out_mp4: str, workdir: str,
                           width=1280, height=720, fps=30, pause_clip_name="pause.mp4",
                           inter_pause=0.12, fallback_spell=True, font_path=None):
    clips_dir = Path(clips_dir)
    out_mp4 = Path(out_mp4)
    wd = ensure_dir(workdir)
    size = (int(width), int(height))

    items = []
    segs = []
    current_time = 0.0
    tokens = text_to_tokens(text)
    # preload normalized pause if exists
    pause_src = clips_dir / pause_clip_name
    pause_norm = None
    if pause_src.exists():
        pause_norm = wd / "pause_norm.mp4"
        normalize_clip_to_mp4(pause_src, pause_norm, size=size, fps=fps)

    idx = 0
    for tok in tokens:
        idx += 1
        # use robust lookup to find the best clip for this token
        match, match_kind = find_clip_for_token(tok, clips_dir)
        if match is not None:
            word_src = match
            # optional debug: print(f"Using clip for token '{tok}' -> {word_src} (kind={match_kind})")
            dst = wd / f"{idx:05d}_word.mp4"
            normalize_clip_to_mp4(word_src, dst, size=size, fps=fps)
            dur = get_video_duration(dst)
            segs.append({"token": tok, "type": "word", "source": str(word_src), "match_kind": match_kind, "start": current_time, "end": current_time + dur})
            items.append(dst)
            current_time += dur
            # add pause
            if pause_norm:
                items.append(pause_norm)
                current_time += get_video_duration(pause_norm)
            else:
                black = wd / f"{idx:05d}_pause.mp4"
                create_black_mp4(black, duration=inter_pause, size=size, fps=fps)
                items.append(black)
                current_time += inter_pause
            continue

        # fallback to finger spelling per letter
        if fallback_spell and len(tok) <= 40:
            for j, ch in enumerate(tok):
                if not ch.isalpha():
                    continue
                finger_src = clips_dir / f"finger_{ch.lower()}.mp4"
                if finger_src.exists():
                    dst = wd / f"{idx:05d}_let_{j}_{ch}.mp4"
                    normalize_clip_to_mp4(finger_src, dst, size=size, fps=fps)
                    dur = get_video_duration(dst)
                    segs.append({"token": ch, "type": "letter", "source": str(finger_src), "start": current_time, "end": current_time + dur})
                    items.append(dst)
                    current_time += dur
                    # tiny pause
                    black = wd / f"{idx:05d}_lpause_{j}.mp4"
                    create_black_mp4(black, duration=0.05, size=size, fps=fps)
                    items.append(black)
                    current_time += 0.05
                else:
                    # generate letter image -> mp4
                    img = make_text_image(ch.upper(), size=size, fontsize=220, font_path=font_path)
                    dst = wd / f"{idx:05d}_letimg_{j}_{ch}.mp4"
                    image_to_mp4(img, dst, duration=0.9, fps=fps, size=size)
                    segs.append({"token": ch, "type": "letter", "source": "generated", "start": current_time, "end": current_time + 0.9})
                    items.append(dst)
                    current_time += 0.9
                    black = wd / f"{idx:05d}_lpause_{j}.mp4"
                    create_black_mp4(black, duration=0.05, size=size, fps=fps)
                    items.append(black)
                    current_time += 0.05
            # small inter-word pause
            black = wd / f"{idx:05d}_wordpause.mp4"
            create_black_mp4(black, duration=inter_pause, size=size, fps=fps)
            items.append(black)
            current_time += inter_pause
            continue

        # final fallback: word card image -> mp4
        img = make_text_image(tok, size=size, fontsize=48, font_path=font_path)
        dst = wd / f"{idx:05d}_card.mp4"
        image_to_mp4(img, dst, duration=1.0, fps=fps, size=size)
        segs.append({"token": tok, "type": "textcard", "source": "generated", "start": current_time, "end": current_time + 1.0})
        items.append(dst)
        current_time += 1.0
        black = wd / f"{idx:05d}_wordpause.mp4"
        create_black_mp4(black, duration=inter_pause, size=size, fps=fps)
        items.append(black)
        current_time += inter_pause

    if not items:
        raise RuntimeError("No items assembled from text")

    # create concat file
    listfile = wd / "concat_list.txt"
    with open(listfile, "w", encoding="utf8") as f:
        for p in items:
            f.write(f"file '{str(Path(p).resolve())}'\n")

    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(out_mp4)
    ]
    run(cmd)
    return segs, str(wd)

# -----------------------
# CLI / Batch
# -----------------------
def process_single(args):
    workdir = args.workdir if args.workdir else f"./tmp_isl_work_{int(time.time())}"
    segs, wd = assemble_text_to_video(args.text, args.clips_dir, args.out, workdir,
                                      width=args.width, height=args.height, fps=args.fps,
                                      pause_clip_name=args.pause_clip, inter_pause=args.inter_clip_pause,
                                      fallback_spell=not args.no_spell, font_path=args.font_path)
    meta = {"text": args.text, "out": args.out, "segments": segs, "workdir": wd}
    print(json.dumps(meta, ensure_ascii=False, indent=2))

def process_batch(args):
    workroot = args.workdir if args.workdir else f"./tmp_isl_batch_{int(time.time())}"
    ensure_dir(workroot)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    with open(args.batch_in, "r", encoding="utf8") as fin, open(manifest, "w", encoding="utf8") as mout:
        for i, line in enumerate(tqdm(fin, desc="batch")):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            uid = obj.get("id") or obj.get("uid") or f"utt_{i}"
            text = obj.get("text") or obj.get("transcript") or ""
            if not text:
                print(f"[WARN] empty text for {uid}, skipping")
                continue
            wd = Path(workroot) / uid
            ensure_dir(wd)
            out_file = out_dir / f"{uid}.mp4"
            segs, _ = assemble_text_to_video(text, args.clips_dir, str(out_file), str(wd),
                                             width=args.width, height=args.height, fps=args.fps,
                                             pause_clip_name=args.pause_clip, inter_pause=args.inter_clip_pause,
                                             fallback_spell=not args.no_spell, font_path=args.font_path)
            meta = {"id": uid, "text": text, "out": str(out_file), "segments": segs}
            mout.write(json.dumps(meta, ensure_ascii=False) + "\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clips_dir", required=True)
    p.add_argument("--text", help="single text to convert")
    p.add_argument("--out", help="output mp4 (single)")
    p.add_argument("--batch_in", help="input jsonl for batch mode")
    p.add_argument("--out_dir", help="output dir for batch mode")
    p.add_argument("--pause_clip", default="pause.mp4")
    p.add_argument("--inter_clip_pause", type=float, default=0.12)
    p.add_argument("--no_spell", action="store_true", help="disable finger-spelling fallback")
    p.add_argument("--workdir", default=None, help="temporary workdir (created if missing)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--font_path", default=None)
    args = p.parse_args()

    if args.batch_in:
        if not args.out_dir:
            raise SystemExit("Please set --out_dir for batch mode")
        process_batch(args)
    else:
        if not args.text or not args.out:
            raise SystemExit("Single mode requires --text and --out")
        process_single(args)

if __name__ == "__main__":
    main()
