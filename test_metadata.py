import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("path_a", type=str)
parser.add_argument("path_b", type=str)
args = parser.parse_args()

with open(Path(args.path_a) / "metadata.json") as f:
    data_a = json.load(f)
with open(Path(args.path_b) / "metadata.json") as f:
    data_b = json.load(f)

keys_a = set(data_a.keys())
keys_b = set(data_b.keys())
missing_keys = keys_a - keys_b
extra_keys = keys_b - keys_a

if missing_keys:
    print(f"Key in A but not in B: {missing_keys}")
if extra_keys:
    print(f"Key in B but not in A: {extra_keys}")
if not missing_keys and not extra_keys:
    print("Keys are the same")
