# test_imports.py
import faulthandler, sys, traceback
faulthandler.enable()
print("PYTHON:", sys.version)

modules = ["datasets", "pyarrow", "soundfile", "torchaudio", "numpy"]
for m in modules:
    try:
        mod = __import__(m)
        print(f"Imported {m} OK, location: {getattr(mod, '__file__', 'builtin')}")
    except Exception as e:
        print(f"FAILED import {m}: {e}")
        traceback.print_exc()
