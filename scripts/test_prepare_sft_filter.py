"""
Quick functional test for the empty-response filter in prepare_sft_data.py.

Constructs a tiny JSONL with 3 normal samples + 2 empty-response samples,
runs prepare, and verifies:
  1. "Skipped 2 samples with empty response" is printed
  2. Output indices contain only 3 samples (empty ones excluded)
  3. All resp_len > 1 in the output

CPU-only, takes ~1s. Run from repo root.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent

# Need a tokenizer with the special tokens prepare expects.
# Reuse whatever the project already ships with, if available.
TOKENIZER_CANDIDATES = [
    REPO_ROOT / "assets" / "tokenizer.json",
    REPO_ROOT / "tokenizer.json",
]


def find_tokenizer() -> Path:
    for p in TOKENIZER_CANDIDATES:
        if p.exists():
            return p
    print("ERROR: could not find a tokenizer.json. Pass one with TOKENIZER=path env var.", file=sys.stderr)
    sys.exit(1)


def main():
    import os
    tokenizer = Path(os.environ.get("TOKENIZER", "")) if os.environ.get("TOKENIZER") else find_tokenizer()
    assert tokenizer.exists(), f"tokenizer not found: {tokenizer}"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        jsonl = tmp / "in.jsonl"
        out = tmp / "out"

        rows = [
            {"instruction": "What is 2+2?", "response": "4"},
            {"instruction": "Capital of France?", "response": "Paris"},
            {"instruction": "Empty response sample A", "response": ""},   # should be dropped
            {"instruction": "Hello?", "response": "Hi there"},
            {"instruction": "Empty response sample B", "response": ""},   # should be dropped
        ]
        with open(jsonl, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        cmd = [
            sys.executable, str(REPO_ROOT / "scripts" / "prepare_sft_data.py"),
            "--train", str(jsonl),
            "--tokenizer", str(tokenizer),
            "--output", str(out),
            "--epochs", "1",
        ]
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("--- stdout ---")
        print(result.stdout)
        print("--- stderr ---")
        print(result.stderr)

        if result.returncode != 0:
            print("FAIL: prepare_sft_data.py exited non-zero")
            sys.exit(1)

        # Check 1: print message
        if "Skipped 2 samples with empty response" not in result.stdout:
            print("FAIL: expected 'Skipped 2 samples with empty response' in stdout")
            sys.exit(1)
        print("OK: skipped count printed correctly")

        # Check 2: output has only 3 samples
        resp_len = np.load(out / "epoch_0" / "resp_len.npy")
        if len(resp_len) != 3:
            print(f"FAIL: expected 3 samples in output, got {len(resp_len)}")
            sys.exit(1)
        print(f"OK: output has {len(resp_len)} samples (empty ones filtered)")

        # Check 3: all resp_len > 1
        if not (resp_len > 1).all():
            print(f"FAIL: some resp_len <= 1 in output: {resp_len.tolist()}")
            sys.exit(1)
        print(f"OK: all resp_len > 1, values: {resp_len.tolist()}")

        print("\nAll checks PASSED.")


if __name__ == "__main__":
    main()
