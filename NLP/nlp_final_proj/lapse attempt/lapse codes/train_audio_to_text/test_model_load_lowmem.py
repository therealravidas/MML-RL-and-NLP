# test_model_load_lowmem.py
import os, sys, faulthandler
faulthandler.enable()
print("PY:", sys.version)
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"

import torch
from transformers import Wav2Vec2ForCTC

print("torch cuda available:", torch.cuda.is_available())
print("Attempting to load model with low_cpu_mem_usage=True ...")
try:
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", low_cpu_mem_usage=True)
    print("Model object created OK")
    # try moving to GPU if available
    if torch.cuda.is_available():
        model.to("cuda")
        print("Moved model to cuda")
    else:
        model.to("cpu")
        print("Moved model to cpu")
    print("SUCCESS: model loaded and moved")
except Exception as e:
    print("FAILED during model load/move:", repr(e))
    raise
