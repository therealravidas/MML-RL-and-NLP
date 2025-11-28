# ================================================
# File: lazy_collator.py
# ================================================

import os
import torchaudio
from transformers import Wav2Vec2Processor
from typing import List, Dict, Any
import torch

class LazyCollator:
    """
    Loads audio lazily per batch from JSON metadata.
    Ensures low memory usage for large datasets.
    """
    def __init__(self, processor: Wav2Vec2Processor, target_sr: int = 16000):
        self.processor = processor
        self.sr = target_sr

    def load_and_resample(self, path: str):
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        if sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.sr)
        return waveform.numpy()

    def __call__(self, features: List[Dict[str, Any]]):
        speech_list = []
        texts = []
        for f in features:
            audio_path = f["audio_path"]
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio missing: {audio_path}")
            speech = self.load_and_resample(audio_path)
            speech_list.append(speech)
            texts.append(f.get("transcript", ""))

        inputs = self.processor(speech_list, sampling_rate=self.sr, return_tensors="pt", padding=True)

        with self.processor.as_target_processor():
            labels = self.processor(texts, padding=True, return_tensors="pt").input_ids

        labels_mask = labels != self.processor.tokenizer.pad_token_id
        labels[~labels_mask] = -100

        return {
            "input_values": inputs.input_values,
            "attention_mask": inputs.attention_mask,
            "labels": labels,
        }
