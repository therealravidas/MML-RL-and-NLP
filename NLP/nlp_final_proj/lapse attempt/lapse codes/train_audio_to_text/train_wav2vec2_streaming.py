#!/usr/bin/env python3
"""
Streaming training for Wav2Vec2 (MLCommons/peoples_speech: dirty_sa)

Usage example:
python train_wav2vec2_streaming.py \
  --repo MLCommons/peoples_speech \
  --config dirty_sa \
  --output_dir ./wav2vec2_stream_out \
  --model_name facebook/wav2vec2-base-960h \
  --per_device_batch_size 2 \
  --grad_accum 8 \
  --epochs 3 \
  --use_cuda

Notes:
- Streaming will NOT save audio to disk. It reads data on demand from the Hub.
- Keep batch sizes very small and use gradient accumulation to avoid memory spikes.
- Run from a terminal and set OMP/MKL/OPENBLAS env vars to 1 to lower native parallelism.
"""
import argparse
import os
import math
import time
import torch
import numpy as np
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from torch.utils.data import DataLoader, IterableDataset
import torchaudio
import soundfile as sf
from typing import Dict, Any, Iterable, List
from pathlib import Path

# -----------------------
# Utilities
# -----------------------
def decode_audio_field(example: Dict[str, Any], default_sr=16000):
    """
    Given a streaming dataset example, try to extract a 1D numpy float32 waveform and sampling rate.
    Handles:
      - example["audio"] as dict with 'array' and 'sampling_rate'
      - example["audio"] as dict with 'path' and possibly 'bytes' (some streaming payloads include array)
      - example["audio_path"] or other keys if present (works if it's a local path)
    Returns (np.ndarray, sr) or (None, None) if not decodeable.
    """
    # 1) prefer 'audio' column if present
    if "audio" in example and example["audio"] is not None:
        a = example["audio"]
        # often a is a dict with keys 'array' and 'sampling_rate'
        if isinstance(a, dict):
            if "array" in a and a["array"] is not None:
                arr = np.asarray(a["array"], dtype="float32")
                if arr.ndim > 1:
                    arr = arr.mean(axis=0)
                sr = a.get("sampling_rate") or a.get("sampling-rate") or default_sr
                return arr.astype("float32"), int(sr)
            # some streaming payloads return 'path' pointing to an http resource — try torchaudio
            if "path" in a and isinstance(a["path"], str):
                path = a["path"]
                try:
                    wav, sr = torchaudio.load(path)  # supports http for many builds
                    wav = wav.mean(dim=0).numpy() if wav.ndim > 1 else wav.squeeze(0).numpy()
                    return wav.astype("float32"), int(sr)
                except Exception:
                    # try soundfile via requests? leave for fallback
                    pass
        # if audio column is already an array or list
        if isinstance(a, list) and len(a) > 0 and isinstance(a[0], (float, int)):
            arr = np.asarray(a, dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            return arr.astype("float32"), default_sr

    # 2) common alternative keys
    for k in ("audio_path", "audio_filepath", "path", "file", "file_path"):
        if k in example and example[k]:
            val = example[k]
            if isinstance(val, str):
                # path could be remote URL (torchaudio supports some http) or local path
                try:
                    wav, sr = torchaudio.load(val)
                    wav = wav.mean(dim=0).numpy() if wav.ndim > 1 else wav.squeeze(0).numpy()
                    return wav.astype("float32"), int(sr)
                except Exception:
                    # if torchaudio fails (e.g., cannot open http), try soundfile if path is bytes-like or local file
                    try:
                        data, sr = sf.read(val, dtype="float32")
                        if data.ndim > 1:
                            data = data.mean(axis=1)
                        return data.astype("float32"), int(sr)
                    except Exception:
                        pass

    # 3) fallback: search for common keys that might contain arrays
    for k in ("samples", "array", "waveform", "audio_array"):
        if k in example and example[k] is not None:
            v = example[k]
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (float, int)):
                arr = np.asarray(v, dtype="float32")
                if arr.ndim > 1:
                    arr = arr.mean(axis=0)
                return arr.astype("float32"), default_sr
            if isinstance(v, dict) and "array" in v:
                arr = np.asarray(v["array"], dtype="float32")
                if arr.ndim > 1:
                    arr = arr.mean(axis=0)
                sr = v.get("sampling_rate") or default_sr
                return arr.astype("float32"), int(sr)

    return None, None

# -----------------------
# Streaming IterableDataset wrapper
# -----------------------
class StreamingHFIterable(IterableDataset):
    """Wraps a HF streaming dataset into a PyTorch IterableDataset that yields raw examples."""
    def __init__(self, dataset_stream, max_examples: int = None):
        self.dataset_stream = dataset_stream
        self.max_examples = max_examples

    def __iter__(self):
        it = iter(self.dataset_stream)
        count = 0
        for ex in it:
            yield ex
            count += 1
            if self.max_examples and count >= self.max_examples:
                break

# -----------------------
# Collator
# -----------------------
class StreamingCollator:
    def __init__(self, processor: Wav2Vec2Processor, target_sr: int = 16000, max_audio_seconds: float = 30.0):
        self.processor = processor
        self.sr = target_sr
        self.max_samples = int(target_sr * max_audio_seconds)

    def __call__(self, batch: List[Dict[str, Any]]):
        speech_list = []
        texts = []
        # decode each example into waveform numpy
        for ex in batch:
            wav, sr = decode_audio_field(ex, default_sr=self.sr)
            if wav is None:
                # skip examples we cannot decode (this keeps the stream flowing)
                continue
            # limit extremely long audio to reduce memory spikes
            if len(wav) > self.max_samples:
                # trim center or start — here trim start
                wav = wav[: self.max_samples]
            # resample if needed
            if sr != self.sr:
                import torch as _torch
                wav_tensor = _torch.from_numpy(wav).unsqueeze(0)  # [1, T]
                wav_tensor = torchaudio.functional.resample(wav_tensor, orig_freq=sr, new_freq=self.sr)
                wav = wav_tensor.squeeze(0).numpy()
            speech_list.append(wav.astype("float32"))
            txt = ex.get("transcript") or ex.get("text") or ex.get("sentence") or ""
            texts.append(txt)

        if len(speech_list) == 0:
            # return empty batch; training loop should handle skip
            return None

        inputs = self.processor(speech_list, sampling_rate=self.sr, return_tensors="pt", padding=True)
        with self.processor.as_target_processor():
            labels = self.processor(texts, padding=True, return_tensors="pt").input_ids

        # convert padding token id's of labels by -100 to ignore in loss
        labels_mask = labels != self.processor.tokenizer.pad_token_id
        labels[~labels_mask] = -100

        batch_out = {
            "input_values": inputs.input_values,
            "attention_mask": inputs.attention_mask,
            "labels": labels,
        }
        return batch_out

# -----------------------
# Training loop
# -----------------------
def train(args):
    # device
    device = "cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu"
    print("Device:", device)

    # processor & model
    print("Loading processor and model:", args.model_name)
    processor = Wav2Vec2Processor.from_pretrained(args.model_name)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_name)
    model.to(device)

    # ensure pad token exists
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(processor.tokenizer))

    # load streaming datasets from hub
    print("Loading streaming datasets from Hub (this will not save audio locally).")
    train_stream = load_dataset(args.repo, args.config, split="train", streaming=True)
    valid_stream = None
    if args.use_validation:
        valid_stream = load_dataset(args.repo, args.config, split="validation", streaming=True)

    # wrap as iterable datasets
    train_iter = StreamingHFIterable(train_stream)
    collator = StreamingCollator(processor, target_sr=args.target_sr, max_audio_seconds=args.max_audio_seconds)

    # DataLoader
    train_loader = DataLoader(train_iter, batch_size=args.per_device_batch_size,
                              collate_fn=collator, num_workers=0)

    # optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps_est = args.estimated_steps or 10000
    if args.use_scheduler:
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps_est)
    else:
        scheduler = None

    model.train()
    global_step = 0
    grad_acc = 0
    running_loss = 0.0

    # training epochs: iterate epoch loops but streaming dataset is continuous; we will break after steps limit
    for epoch in range(args.epochs):
        print(f"=== Epoch {epoch+1}/{args.epochs} ===")
        for batch_idx, batch in enumerate(train_loader):
            # collator may return None if all examples decoded failed
            if batch is None:
                continue

            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            outputs = model(input_values, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss_val = loss.item()
            loss = loss / args.grad_accum
            loss.backward()
            grad_acc += 1

            if grad_acc >= args.grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                grad_acc = 0
                global_step += 1
                running_loss += loss_val

                if global_step % args.log_every == 0:
                    avg_loss = running_loss / max(1, args.log_every)
                    print(f"[step {global_step}] avg_loss={avg_loss:.4f} lr={optimizer.param_groups[0]['lr']:.3e}")
                    running_loss = 0.0

                if global_step % args.save_every == 0:
                    out_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(out_dir)
                    processor.save_pretrained(out_dir)
                    print("Saved checkpoint to", out_dir)

                # optional eval
                if args.use_validation and (global_step % args.eval_every == 0):
                    evaluate_streaming(model, processor, args, valid_stream, device)

                # stop if reached maximum steps
                if args.max_steps and global_step >= args.max_steps:
                    print("Reached max_steps", args.max_steps)
                    return

    # final save
    model.save_pretrained(Path(args.output_dir) / "final")
    processor.save_pretrained(Path(args.output_dir) / "final")
    print("Training complete. Model saved to", args.output_dir)

# -----------------------
# Simple streaming evaluation (limited number of examples)
# -----------------------
def evaluate_streaming(model, processor, args, valid_stream, device):
    if valid_stream is None:
        print("No validation stream provided.")
        return
    print("Running streaming evaluation (limited examples)...")
    model.eval()
    # build small iterable and collate like in training
    valid_iter = StreamingHFIterable(valid_stream, max_examples=args.eval_steps_per_epoch * args.per_device_batch_size)
    collator = StreamingCollator(processor, target_sr=args.target_sr, max_audio_seconds=args.max_audio_seconds)
    loader = DataLoader(valid_iter, batch_size=args.per_device_batch_size, collate_fn=collator, num_workers=0)

    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            out = model(input_values, attention_mask=attention_mask, labels=labels)
            total_loss += out.loss.item()
            n_batches += 1
            if n_batches >= args.eval_steps_per_epoch:
                break
    avg_loss = total_loss / max(1, n_batches)
    print(f"[eval] avg_loss={avg_loss:.4f} over {n_batches} batches")
    model.train()

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, default="MLCommons/peoples_speech", help="dataset repo on HF")
    parser.add_argument("--config", type=str, default="dirty_sa", help="dataset config (split variant)")
    parser.add_argument("--output_dir", type=str, default="./wav2vec2_stream_out")
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base-960h")
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--target_sr", type=int, default=16000)
    parser.add_argument("--max_audio_seconds", type=float, default=30.0)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--use_validation", action="store_true")
    parser.add_argument("--eval_steps_per_epoch", type=int, default=20)
    parser.add_argument("--estimated_steps", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=None, help="stop after this many optimizer updates (global steps)")
    parser.add_argument("--use_scheduler", action="store_true")
    args = parser.parse_args()

    Path = Path if 'Path' in globals() else None
    Path = __import__('pathlib').Path

    os.makedirs(args.output_dir, exist_ok=True)
    # run training
    train(args)
