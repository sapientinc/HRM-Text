import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file
from transformers import AutoTokenizer

from dataset_new import V1DatasetMeta
from pretrain import PretrainConfig
from simple_inference_engine import inference_load_checkpoint


SKIP_PREFIXES = (
    "model.H_level.core.rotary_emb.",
    "model.L_level.core.rotary_emb.",
)
DROP_KEYS = {"model.zH_init"}


def remap_key(key: str) -> str | None:
    if key in DROP_KEYS or key.startswith(SKIP_PREFIXES):
        return None
    key = key.replace("model.H_level.core.layers.", "model.H_module.layers.")
    key = key.replace("model.L_level.core.layers.", "model.L_module.layers.")
    key = key.replace("model.zL_init", "model.z_L_init")
    return "model.embed_tokens.weight" if key == "embed_tokens.embedding_weight" else key


def convert_state_dict(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], list[str]]:
    out, skipped = {}, []
    for key, value in state_dict.items():
        new_key = remap_key(key)
        if new_key is None:
            skipped.append(key)
        else:
            out[new_key] = value.contiguous()
    return out, skipped


def _compute_intermediate_size(hidden_size: int, expansion: float) -> int:
    return ((round(expansion * hidden_size * 2 / 3) + 255) // 256) * 256


def _compute_l_bp_steps(cfg: dict) -> list[int]:
    H, L = int(cfg["H_cycles"]), int(cfg["L_cycles"])
    bp_steps = int(cfg.get("bp_max_steps", cfg.get("max_bp_steps", H + 1)))
    h_bp_steps = min(H, max(0, bp_steps - 1))
    l_bp_steps = min(H * L, max(0, bp_steps - h_bp_steps))
    threshold = H * L - l_bp_steps
    return [max(0, min(L, (i + 1) * L - threshold)) for i in range(H)]


def load_config(ckpt_path: Path) -> tuple[V1DatasetMeta, dict]:
    model_cfg = PretrainConfig(**yaml.safe_load((ckpt_path / "all_config.yaml").read_text()))
    metadata = V1DatasetMeta(**yaml.safe_load((ckpt_path / "train_metadata.yaml").read_text()))
    return metadata, model_cfg.arch.model_dump() | metadata.model_dump() | model_cfg.data.model_dump()


def build_hf_config(cfg: dict, tokenizer) -> dict:
    hidden_size = cfg["hidden_size"]
    init_type, init_std = cfg.get("init_type", "fixed_normal"), cfg.get("init_std")
    if init_type == "lecun_normal":
        in_std = 1.0 / math.sqrt(hidden_size)
    elif init_std is not None:
        in_std = init_std
    else:
        in_std = 1.0 / math.sqrt(hidden_size) if init_type == "megatron" else 0.02

    hf_cfg = {
        "model_type": "hrm_text",
        "architectures": ["HrmTextForCausalLM"],
        "vocab_size": cfg["vocab_size"],
        "hidden_size": hidden_size,
        "intermediate_size": _compute_intermediate_size(hidden_size, cfg.get("expansion", 4.0)),
        "num_hidden_layers": cfg["n_layers"],
        "num_attention_heads": cfg["num_heads"],
        "num_key_value_heads": cfg["num_heads"],
        "head_dim": hidden_size // cfg["num_heads"],
        "H_cycles": cfg["H_cycles"],
        "L_cycles": cfg["L_cycles"],
        "L_bp_steps": _compute_l_bp_steps(cfg),
        "max_position_embeddings": cfg["max_seq_len"],
        "rms_norm_eps": cfg.get("norm_eps", 1e-6),
        "rope_theta": cfg.get("rope_theta", 10000.0),
        "tie_word_embeddings": False,
        "initializer_range": in_std,
        "embedding_scale": 1.0 / in_std,
        "prefix_lm": True,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None) or 0,
    }
    for key, token_name in (("bos_token_id", "boq"), ("eos_token_id", "eoa")):
        if token_name in cfg:
            hf_cfg[key] = tokenizer.convert_tokens_to_ids(cfg[token_name])
    return {k: v for k, v in hf_cfg.items() if v is not None}


def tokenizer_path(metadata: V1DatasetMeta, override: Path | None) -> Path:
    path = override or Path(metadata.tokenizer_info["tokenizer_path"])
    return path.parent if path.name == "tokenizer.json" else path


def set_tokenizer_special_tokens(tokenizer, cfg: dict):
    if "boq" in cfg:
        tokenizer.bos_token = cfg["boq"]
    if "eoa" in cfg:
        tokenizer.eos_token = cfg["eoa"]
    return tokenizer


def parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--ckpt_epoch", type=int, default=None)
    parser.add_argument("--ckpt_use_ema", type=parse_bool, default=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--tokenizer_path", type=Path, default=None)
    args = parser.parse_args()

    metadata, cfg = load_config(args.ckpt_path)
    ckpt = inference_load_checkpoint(str(args.ckpt_path), args.ckpt_epoch, args.ckpt_use_ema)

    hf_state, dropped = convert_state_dict(ckpt.model.state_dict())
    print(f"[convert] mapped {len(hf_state)} tensors; dropped {len(dropped)}")

    tok_path = tokenizer_path(metadata, args.tokenizer_path)
    print(f"[convert] using tokenizer at {tok_path}")
    tokenizer = ckpt.tokenizer if args.tokenizer_path is None else AutoTokenizer.from_pretrained(str(tok_path), use_fast=True)
    tokenizer = set_tokenizer_special_tokens(tokenizer, metadata.tokenizer_info)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(json.dumps(build_hf_config(cfg | metadata.tokenizer_info, tokenizer), indent=2))
    save_file(hf_state, args.out_dir / "model.safetensors")
    tokenizer.save_pretrained(args.out_dir)
    print(f"[convert] wrote checkpoint to {args.out_dir}")


if __name__ == "__main__":
    main()
