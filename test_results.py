import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("path_a", type=str)
parser.add_argument("path_b", type=str)
args = parser.parse_args()

with open(Path(args.path_a) / "results.jsonl") as f:
    data_a = [json.loads(line) for line in f]
with open(Path(args.path_b) / "results.jsonl") as f:
    data_b = [json.loads(line) for line in f]

row_a = data_a[0]
row_b = data_b[0]
keys_a = set(row_a.keys())
keys_b = set(row_b.keys())
missing_keys = keys_a - keys_b
extra_keys = keys_b - keys_a

if missing_keys:
    print(f"Key in A but not in B: {missing_keys}")
if extra_keys:
    print(f"Key in B but not in A: {extra_keys}")
if not missing_keys and not extra_keys:
    print("Keys are the same")
