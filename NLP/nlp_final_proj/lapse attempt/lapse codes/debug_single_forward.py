#!/usr/bin/env python3
import os, sys, faulthandler, psutil, time, json
faulthandler.enable()
print("PY:", sys.version)
print("PID:", os.getpid())
print("cwd:", os.getcwd())

# === CONFIG - change only if needed ===
JSONL = "/home/BTECH_7TH_SEM/pspeech_meta_jsonl/train.resolved.jsonl"   # <--- adjust if different
SAMPLE_COUNT = 5   # small number to scan for an existing audio file
MODEL_NAME = "facebook/wav2vec2-base-960h"
USE_CUDA = True    # set False to force CPU version
# ======================================

def memprint(tag=""):
    vm = psutil.virtual_memory()
    print(f"[{tag}] mem: total={vm.total//(1024**3)}G used={vm.used//(1024**3)}G free={vm.available//(1024**3)}G swap_free={(psutil.swap_memory().free)//(1024**3)}G")

print("Environment thread limits:",
      os.environ.get("OMP_NUM_THREADS"),
      os.environ.get("MKL_NUM_THREADS"),
      os.environ.get("OPENBLAS_NUM_THREADS"))

memprint("start")

# 1) load small metadata slice
from datasets import load_dataset
print("Loading small metadata slice from:", JSONL)
try:
    ds = load_dataset("json", data_files={"train": JSONL})["train"].select(range(SAMPLE_COUNT))
    print("Loaded metadata rows:", len(ds))
except Exception as e:
    print("Failed to load JSONL:", e)
    sys.exit(1)

# 2) find a valid audio path that exists
audio_path = None
for ex in ds:
    for k in ("audio_path","audio","path","file","file_path","audio_filepath"):
        if k in ex and ex[k]:
            # if dict with path
            v = ex[k]
            if isinstance(v, dict) and "path" in v:
                cand = v["path"]
            else:
                cand = v if isinstance(v, str) else None
            if cand and os.path.exists(cand):
                audio_path = cand
                break
    if audio_path: break

print("Chosen audio:", audio_path)
if not audio_path:
    print("No existing audio file found in the small sample. Exiting.")
    sys.exit(1)

memprint("before_load_audio")
# 3) load audio with torchaudio
import torchaudio
try:
    waveform, sr = torchaudio.load(audio_path)
    print("torchaudio loaded shape:", waveform.shape, "sr:", sr)
except Exception as e:
    print("torchaudio.load failed:", e)
    sys.exit(1)

memprint("after_load_audio")

# 4) resample (if needed)
if sr != 16000:
    import torch
    try:
        t = waveform
        if t.shape[0] > 1:
            t = t.mean(dim=0, keepdim=True)
        t = t.squeeze(0).unsqueeze(0)  # shape [1, T]
        memprint("before_resample")
        t = torchaudio.functional.resample(t, orig_freq=sr, new_freq=16000)
        print("resampled shape:", t.shape)
        waveform = t.squeeze(0)
    except Exception as e:
        print("Resample failed:", e)
        sys.exit(1)

memprint("after_resample")

# 5) load processor and model
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
print("Loading processor:", MODEL_NAME)
try:
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    print("Processor loaded.")
except Exception as e:
    print("Processor load failed:", e)
    sys.exit(1)

memprint("after_processor")

print("Loading model:", MODEL_NAME, "USE_CUDA=", USE_CUDA)
try:
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    device = "cuda" if (USE_CUDA and torch.cuda.is_available()) else "cpu"
    model.to(device)
    print("Model loaded to", device)
except Exception as e:
    print("Model load/move failed:", e)
    sys.exit(1)

memprint("after_model_load")

# 6) create inputs and run one forward
with torch.no_grad():
    try:
        wav_np = waveform.numpy() if hasattr(waveform, "numpy") else waveform
        inputs = processor([wav_np], sampling_rate=16000, return_tensors="pt", padding=True)
        # move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        memprint("before_forward")
        out = model(inputs["input_values"])
        memprint("after_forward")
        print("Output keys:", out.keys())
    except Exception as e:
        print("Forward failed:", e)
        sys.exit(1)

print("SUCCESS: single forward completed.")
