import torch
from torch import Tensor

from models.flash_attention_prefixlm_common import prefixlm_seq_info_from_tensors, prefixlm_sequence_indices

__all__ = ["flash_attn_varlen_prefixlm"]


def _rocm_varlen(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
) -> Tensor:
    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError as exc:
        raise ImportError(
            "accelerator_type=rocm requires the ROCm FlashAttention 2 build exposing flash_attn.flash_attn_varlen_func."
        ) from exc

    return flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=causal,
    )


def flash_attn_varlen_prefixlm(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool,
    prefix_lens: Tensor,
    causal_lens: Tensor,
    cu_seqlens: Tensor,
    total_seqlen: Tensor,
    numseqs: Tensor,
    max_seqlen_prefix: Tensor,
    max_seqlen_causal: Tensor,
    max_seqlen_all: Tensor,
) -> Tensor:
    info = prefixlm_seq_info_from_tensors(
        prefix_lens,
        causal_lens,
        cu_seqlens,
        total_seqlen,
        numseqs,
        max_seqlen_prefix,
        max_seqlen_causal,
        max_seqlen_all,
    )
    prefix_cu_seqlens = torch.nn.functional.pad(torch.cumsum(info.prefix_lens, dim=0, dtype=torch.int32), (1, 0))
    seq_idx, token_idx, valid_idx = prefixlm_sequence_indices(info)
    mask = token_idx < info.prefix_lens[seq_idx]

    out = torch.zeros_like(q)
    prefix_idx = valid_idx[mask]
    out_bidir = _rocm_varlen(
        q[prefix_idx],
        k[prefix_idx],
        v[prefix_idx],
        cu_seqlens_q=prefix_cu_seqlens,
        cu_seqlens_k=prefix_cu_seqlens,
        max_seqlen_q=info.max_seqlen_prefix,
        max_seqlen_k=info.max_seqlen_prefix,
        causal=is_causal,
    )
    out[prefix_idx] = out_bidir

    active = info.causal_lens > 0
    causal_idx = valid_idx[(~mask) & active[seq_idx]]
    if causal_idx.numel() > 0:
        total_lens = info.prefix_lens + info.causal_lens
        active_total_lens = total_lens[active]
        active_causal_lens = info.causal_lens[active]
        active_key_idx = valid_idx[active[seq_idx]]
        active_cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(active_total_lens, dim=0, dtype=torch.int32), (1, 0))
        active_cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(active_causal_lens, dim=0, dtype=torch.int32), (1, 0))

        out_causal = _rocm_varlen(
            q[causal_idx],
            k[active_key_idx],
            v[active_key_idx],
            cu_seqlens_q=active_cu_seqlens_q,
            cu_seqlens_k=active_cu_seqlens_k,
            max_seqlen_q=int(active_causal_lens.max().item()),
            max_seqlen_k=int(active_total_lens.max().item()),
            causal=True,
        )
        out[causal_idx] = out_causal

    return out
