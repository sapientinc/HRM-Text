"""Prepare SFT data for HRM-Text fine-tuning.

Converts a generic JSONL into the V1Dataset binary layout that
`dataset_new.py` consumes directly. Prompt construction (few-shot demos,
schema injection, task framing, etc.) is the caller's responsibility —
the `instruction` field is tokenized verbatim.

Input JSONL (one object per line):
    {"instruction": "<full prompt>", "response": "<expected output>",
     "condition": "direct"}      # condition optional; defaults to "direct"

Output directory:
    <out>/metadata.json
    <out>/tokens.npy
    <out>/tokenizer_info.json
    <out>/tokenizer.json                   # copy for self-containment
    <out>/epoch_0/{inst_start,inst_len,resp_start,resp_len}.npy
    <out>/epoch_1/...
    ...

Usage:
    python scripts/prepare_sft_data.py \
        --train my_sft.jsonl \
        --tokenizer /path/to/tokenizer.json \
        --output /tmp/sft_data \
        --epochs 10
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Input JSONL.")
    ap.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    ap.add_argument("--output", required=True, help="Output directory.")
    ap.add_argument("--epochs", type=int, required=True,
                    help="Pre-compute N epoch shuffles (must match training epochs).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--context-size", type=int, default=4097,
                    help="Must be at least max sample length + 1 (AR shift).")
    ap.add_argument("--boq", default="<|im_start|>")
    ap.add_argument("--eoq", default="<|im_end|>")
    ap.add_argument("--eoa", default="<|box_end|>")
    ap.add_argument(
        "--conditions",
        default="direct=<|object_ref_start|>,cot=<|object_ref_end|>,noisy=<|quad_start|>,synth=<|quad_end|>",
        help="Comma-separated key=token pairs mapping condition labels to vocab tokens.",
    )
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer.from_file(args.tokenizer)

    def _id(name: str) -> int:
        tid = tok.token_to_id(name)
        if tid is None:
            raise ValueError(f"special token {name!r} not in tokenizer vocab")
        return tid

    boq_id, eoq_id, eoa_id = _id(args.boq), _id(args.eoq), _id(args.eoa)

    cond_map: dict[str, int] = {}
    cond_mapping_tokens: dict[str, str] = {}
    for pair in args.conditions.split(","):
        k, v = pair.split("=")
        cond_map[k] = _id(v)
        cond_mapping_tokens[k] = v

    with open(args.train, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(rows)} samples from {args.train}")

    all_tokens: list[int] = []
    inst_start: list[int] = []
    inst_len: list[int] = []
    resp_start: list[int] = []
    resp_len: list[int] = []
    skipped_empty = 0

    for r in rows:
        condition = r.get("condition", "direct")
        if condition not in cond_map:
            raise ValueError(
                f"sample condition {condition!r} not in --conditions map "
                f"(known: {sorted(cond_map)})"
            )

        inst_ids = tok.encode(r["instruction"], add_special_tokens=False).ids
        resp_ids = tok.encode(r["response"], add_special_tokens=False).ids

        # Empty-response samples produce seqlens_o == 0 at runtime; a packed batch of
        # only such samples crashes FA3 backward. Drop them at prep time.
        if not resp_ids:
            skipped_empty += 1
            continue

        i_start = len(all_tokens)
        all_tokens.append(boq_id)
        all_tokens.append(cond_map[condition])
        all_tokens.extend(inst_ids)
        all_tokens.append(eoq_id)
        inst_start.append(i_start)
        inst_len.append(len(all_tokens) - i_start)

        r_start = len(all_tokens)
        all_tokens.extend(resp_ids)
        all_tokens.append(eoa_id)
        resp_start.append(r_start)
        resp_len.append(len(all_tokens) - r_start)

    if skipped_empty:
        print(f"Skipped {skipped_empty} samples with empty response")

    tokens_np = np.array(all_tokens, dtype=np.int32)
    inst_start_np = np.array(inst_start, dtype=np.int64)
    inst_len_np = np.array(inst_len, dtype=np.int64)
    resp_start_np = np.array(resp_start, dtype=np.int64)
    resp_len_np = np.array(resp_len, dtype=np.int64)

    sample_lens = inst_len_np + resp_len_np
    max_len = int(sample_lens.max())
    if max_len >= args.context_size:
        raise ValueError(
            f"longest sample is {max_len} tokens but --context-size is "
            f"{args.context_size}; bump --context-size."
        )
    print(f"Tokens: {len(all_tokens):,}  avg sample = {sample_lens.mean():.1f}  max = {max_len}")

    np.save(out_dir / "tokens.npy", tokens_np)

    # Self-contained tokenizer copy so downstream tools (inference / convert_to_hf)
    # can find it without needing the original path.
    shutil.copyfile(args.tokenizer, out_dir / "tokenizer.json")

    tokenizer_info = {
        "tokenizer_path": str(out_dir),
        "boq": args.boq,
        "eoq": args.eoq,
        "eoa": args.eoa,
        "condition_mapping": cond_mapping_tokens,
        "vocab_size": tok.get_vocab_size(with_added_tokens=True),
    }
    with open(out_dir / "tokenizer_info.json", "w") as f:
        json.dump(tokenizer_info, f)

    # vocab_size MUST be None at the meta level; the training loop derives the
    # padded vocab from the model arch (see dataset_new.py).
    meta = {
        "tokenizer_info": tokenizer_info,
        "vocab_size": None,
        "max_seq_len": args.context_size,
        "total_length": int(sample_lens.sum()),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f)

    rng = np.random.Generator(np.random.Philox(seed=args.seed))
    for epoch in range(args.epochs):
        perm = rng.permutation(len(inst_start_np))
        ep_dir = out_dir / f"epoch_{epoch}"
        ep_dir.mkdir(exist_ok=True)
        np.save(ep_dir / "inst_start.npy", inst_start_np[perm])
        np.save(ep_dir / "inst_len.npy", inst_len_np[perm])
        np.save(ep_dir / "resp_start.npy", resp_start_np[perm])
        np.save(ep_dir / "resp_len.npy", resp_len_np[perm])

    print(f"Wrote {args.epochs} epoch shuffles to {out_dir}")
    print(f"Done. Point cfg.data.path to: {out_dir}")


if __name__ == "__main__":
    main()
