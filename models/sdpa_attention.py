"""PyTorch SDPA-based replacement for the FlashAttention-3 kernels in
`flash_attention_prefixlm_v2.py` and the FA3 `flash_attn_with_kvcache` API.

FA3's "hopper" kernels are hand-written for sm_90 (Hopper) WGMMA/TMA
instructions and do not build or run on other architectures (e.g. Blackwell
sm_121 on DGX Spark). This module reproduces the same PrefixLM attention
semantics using `torch.nn.functional.scaled_dot_product_attention`, so it
runs anywhere PyTorch's CUDA/CPU SDPA backends do. It is not a fused kernel
and materializes an attention mask, so it is slower and more memory-hungry
than FA3 for very long, densely packed sequences -- fine for small-scale
validation runs, but treat it as a correctness-first fallback rather than
a performance-equivalent replacement.

`models/layers.py` imports the FA3 versions first and only falls back to
this module on ImportError, so nothing changes for environments where FA3
is installed (e.g. Hopper cloud nodes).
"""

from typing import Union

import torch
import torch.nn.functional as F
from torch import Tensor


def _to_int(x: Union[int, Tensor]) -> int:
    return int(x.item()) if torch.is_tensor(x) else int(x)


def _build_prefixlm_mask(prefix_lens: Tensor, cu_seqlens: Tensor, numseqs: int, total_seqlen: int,
                          total_len: int, is_causal: bool, device: torch.device) -> Tensor:
    """Boolean allow-mask [total_len, total_len] (True = attend), replicating the
    two FA3 passes in `flash_attention_prefixlm_v2.py`:

    - "prefix" tokens (the first `prefix_lens[s]` tokens of packed sequence s)
      attend to other prefix tokens of the same sequence, bidirectionally
      (or causally among themselves if `is_causal`).
    - remaining "causal" tokens of sequence s always attend causally to every
      token (prefix + causal) at or before their own position in that sequence.
    - tokens never attend across packed-sequence boundaries.
    - padding beyond `total_seqlen` is masked to at least self-attend, so no
      row is fully disallowed (the caller zeroes padded outputs afterward).
    """
    cu = cu_seqlens[:numseqs + 1].to(device=device, dtype=torch.long)
    starts = cu[:-1]
    ends = cu[1:]
    prefix = prefix_lens[:numseqs].to(device=device, dtype=torch.long)

    idx = torch.arange(total_len, device=device)
    valid = idx < total_seqlen

    # Index of the packed sequence containing `idx` (clamped so gathers stay
    # in-bounds for padding positions; those results are discarded via `valid`).
    seq_of_tok = torch.bucketize(idx, ends, right=True).clamp(max=max(numseqs - 1, 0))
    seq_start = starts[seq_of_tok]
    prefix_len_tok = prefix[seq_of_tok]

    local_pos = torch.where(valid, idx - seq_start, idx)  # padding: fall back to absolute index (still monotonic -> self always visible)
    is_prefix = valid & (local_pos < torch.where(valid, prefix_len_tok, torch.zeros_like(idx)))
    seq_id = torch.where(valid, seq_of_tok, torch.full_like(seq_of_tok, -1))

    same_seq = seq_id.unsqueeze(1) == seq_id.unsqueeze(0)           # [Lq, Lk]
    causal_ok = local_pos.unsqueeze(0) <= local_pos.unsqueeze(1)     # key pos <= query pos
    key_is_prefix = is_prefix.unsqueeze(0)
    query_is_prefix = is_prefix.unsqueeze(1)

    prefix_rule = key_is_prefix & causal_ok if is_causal else key_is_prefix
    allow = same_seq & torch.where(query_is_prefix, prefix_rule, causal_ok)
    return allow


def flash_attn_varlen_prefixlm_sdpa(q: Tensor, k: Tensor, v: Tensor, is_causal: bool,
                                    prefix_lens: Tensor, causal_lens: Tensor, cu_seqlens: Tensor,
                                    total_seqlen: Union[int, Tensor], numseqs: Union[int, Tensor],
                                    max_seqlen_prefix: Union[int, Tensor], max_seqlen_causal: Union[int, Tensor],
                                    max_seqlen_all: Union[int, Tensor]) -> Tensor:
    # q: [L, Hq, D], k/v: [L, Hkv, D] -- packed varlen, no batch dim.
    del causal_lens, max_seqlen_prefix, max_seqlen_causal, max_seqlen_all  # implied by prefix_lens/cu_seqlens
    total_len, num_q_heads, _ = q.shape
    num_kv_heads = k.shape[1]
    numseqs_i = _to_int(numseqs)
    total_seqlen_i = _to_int(total_seqlen)

    allow = _build_prefixlm_mask(prefix_lens, cu_seqlens, numseqs_i, total_seqlen_i, total_len, is_causal, q.device)

    q_ = q.transpose(0, 1).unsqueeze(0)  # [1, Hq, L, D]
    k_ = k.transpose(0, 1).unsqueeze(0)  # [1, Hkv, L, D]
    v_ = v.transpose(0, 1).unsqueeze(0)
    if num_q_heads != num_kv_heads:
        rep = num_q_heads // num_kv_heads
        k_ = k_.repeat_interleave(rep, dim=1)
        v_ = v_.repeat_interleave(rep, dim=1)

    out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=allow.unsqueeze(0).unsqueeze(0))
    out = out.squeeze(0).transpose(0, 1)  # [L, Hq, D]
    if total_seqlen_i < total_len:
        # Out-of-place: an in-place slice-assign here would mutate an autograd-tracked
        # SDPA output tensor and break the backward pass (version-counter mismatch).
        valid = (torch.arange(total_len, device=out.device) < total_seqlen_i).view(-1, 1, 1)
        out = out * valid.to(out.dtype)  # match FA3's zero-padding of unused tail
    return out


def sdpa_attn_with_kvcache(q: Tensor, k: Tensor, v: Tensor, k_cache: Tensor, v_cache: Tensor,
                           cache_seqlens: Union[int, Tensor], causal: bool, num_splits: int = 1) -> Tensor:
    # q/k/v: [B, S_new, H, D]. k_cache/v_cache: [B, max_seq_len, Hkv, D], updated in place.
    del num_splits
    batch_size, seqlen_new, num_q_heads, _ = q.shape
    num_kv_heads = k.shape[2]
    device = q.device

    if not torch.is_tensor(cache_seqlens):
        cache_seqlens = torch.full((batch_size,), cache_seqlens, dtype=torch.long, device=device)
    cache_seqlens = cache_seqlens.to(device=device, dtype=torch.long)

    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)
    new_pos = cache_seqlens.unsqueeze(1) + torch.arange(seqlen_new, device=device).unsqueeze(0)
    k_cache[batch_idx, new_pos] = k
    v_cache[batch_idx, new_pos] = v

    max_len = int((cache_seqlens.max() + seqlen_new).item())
    k_all = k_cache[:, :max_len]
    v_all = v_cache[:, :max_len]
    if num_q_heads != num_kv_heads:
        rep = num_q_heads // num_kv_heads
        k_all = k_all.repeat_interleave(rep, dim=2)
        v_all = v_all.repeat_interleave(rep, dim=2)

    key_pos = torch.arange(max_len, device=device).view(1, 1, max_len)
    q_abs_pos = cache_seqlens.view(batch_size, 1, 1) + torch.arange(seqlen_new, device=device).view(1, seqlen_new, 1)
    if causal:
        allow = key_pos <= q_abs_pos
    else:
        valid_len = cache_seqlens.view(batch_size, 1, 1) + seqlen_new
        allow = key_pos < valid_len
    allow = allow.unsqueeze(1)  # [B, 1, S_new, max_len], broadcasts over heads

    q_ = q.transpose(1, 2)      # [B, Hq, S_new, D]
    k_ = k_all.transpose(1, 2)  # [B, Hq, max_len, D]
    v_ = v_all.transpose(1, 2)
    out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=allow)
    return out.transpose(1, 2)  # [B, S_new, Hq, D]
