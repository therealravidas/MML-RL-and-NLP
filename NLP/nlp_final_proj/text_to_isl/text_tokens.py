#!/usr/bin/env python3
# test_clip_mapping.py
import re, argparse, json
from pathlib import Path
from difflib import get_close_matches

def token_to_filename(token: str) -> str:
    t = token.strip().lower()
    t = re.sub(r"[^a-z0-9]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t

def find_match(token, clips_dir):
    token_name = token_to_filename(token)
    # 1) exact filename
    p = clips_dir / f"{token_name}.mp4"
    if p.exists(): return ("exact", p)
    # 2) token appears as substring in any filename (case-insensitive)
    for f in clips_dir.iterdir():
        if not f.is_file(): continue
        n = f.name.lower()
        if token_name in n or token.lower() in n:
            return ("substring", f)
    # 3) close fuzzy match using difflib on basenames
    basenames = [f.name for f in clips_dir.iterdir() if f.is_file()]
    matches = get_close_matches(token_name + ".mp4", basenames, n=1, cutoff=0.6)
    if matches:
        return ("fuzzy", clips_dir / matches[0])
    return (None, None)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--clips_dir", required=True)
    p.add_argument("--text", help="single sentence to test (optional)")
    p.add_argument("--jsonl", help="jsonl file with text or transcript field (optional)")
    p.add_argument("--top", type=int, default=20, help="how many sample tokens to show")
    args = p.parse_args()

    clips_dir = Path(args.clips_dir)
    if not clips_dir.exists():
        raise SystemExit("clips_dir missing: " + str(clips_dir))

    tokens = []
    if args.text:
        tokens = re.findall(r"[A-Za-z0-9]+", args.text)
    elif args.jsonl:
        import json
        with open(args.jsonl, "r", encoding="utf8") as f:
            for i,l in enumerate(f):
                if i>1000: break
                obj = json.loads(l)
                t = obj.get("text") or obj.get("transcript") or ""
                tokens += re.findall(r"[A-Za-z0-9]+", t)
    else:
        raise SystemExit("provide --text or --jsonl")

    tokens = list(dict.fromkeys(tokens))  # dedupe, preserve order
    from pathlib import Path
    print(f"clips_dir: {clips_dir} ; total clip files: {len(list(clips_dir.iterdir()))}")
    report = []
    for tok in tokens[:args.top]:
        kind, found = find_match(tok, clips_dir)
        report.append({"token": tok, "normalized": token_to_filename(tok), "match_kind": kind, "match_path": str(found) if found else None})
    print(json.dumps(report, indent=2))
    # show unmatched tokens
    unmatched = [r for r in report if r["match_kind"] is None]
    if unmatched:
        print("\nUNMATCHED TOKENS (first 20):")
        for u in unmatched[:20]:
            print(" ", u["token"], "-> expected filename:", u["normalized"] + ".mp4")
    else:
        print("\nAll sample tokens matched a clip (good).")
