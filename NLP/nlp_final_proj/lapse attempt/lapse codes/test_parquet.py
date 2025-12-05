# test_parquet.py
import faulthandler, sys
faulthandler.enable()
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
PARQUET_PATH = "/home/BTECH_7TH_SEM/Desktop/NLP Datasets/peoples_speech_dirty_sa/dirty_sa/test-00000-of-00008.parquet"  # change as needed

print("PYTHON:", sys.version)
p = Path(PARQUET_PATH)
print("exists:", p.exists(), "size:", p.stat().st_size)
tbl = pq.read_table(str(p), columns=None)
print("Table schema:", tbl.schema)
print("Rows:", tbl.num_rows)
print("First row example:", tbl.to_pydict() and {k: v[0] for k, v in tbl.to_pydict().items()})
