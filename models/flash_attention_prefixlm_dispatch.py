import torch
from torch import Tensor

from models.accelerator import get_accelerator_type

__all__ = ["flash_attn_varlen_prefixlm"]


def _mps_kernel_supported(q: Tensor, k: Tensor, v: Tensor) -> bool:
    return q.device.type == "mps" and q.dtype == torch.float32 and q.shape == k.shape == v.shape


def _mps_prefixlm(
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
    from models.flash_attention_prefixlm_mps import flash_attn_varlen_prefixlm_mps

    return flash_attn_varlen_prefixlm_mps(
        q,
        k,
        v,
        is_causal,
        prefix_lens,
        causal_lens,
        cu_seqlens,
        total_seqlen,
        numseqs,
        max_seqlen_prefix,
        max_seqlen_causal,
        max_seqlen_all,
    )


def _dense_prefixlm(
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
    from models.flash_attention_prefixlm_dense import flash_attn_varlen_prefixlm as dense_prefixlm

    return dense_prefixlm(
        q,
        k,
        v,
        is_causal,
        prefix_lens,
        causal_lens,
        cu_seqlens,
        total_seqlen,
        numseqs,
        max_seqlen_prefix,
        max_seqlen_causal,
        max_seqlen_all,
    )


@torch.compiler.disable
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
    match get_accelerator_type():
        case "sm90":
            raise RuntimeError("SM90 PrefixLM attention is implemented in models.flash_attention_prefixlm_v2.")
        case "sm100":
            from models.flash_attention_prefixlm_fa4 import flash_attn_varlen_prefixlm as fa4_prefixlm

            return fa4_prefixlm(
                q,
                k,
                v,
                is_causal,
                prefix_lens,
                causal_lens,
                cu_seqlens,
                total_seqlen,
                numseqs,
                max_seqlen_prefix,
                max_seqlen_causal,
                max_seqlen_all,
            )
        case "rocm":
            from models.flash_attention_prefixlm_rocm import flash_attn_varlen_prefixlm as rocm_prefixlm

            return rocm_prefixlm(
                q,
                k,
                v,
                is_causal,
                prefix_lens,
                causal_lens,
                cu_seqlens,
                total_seqlen,
                numseqs,
                max_seqlen_prefix,
                max_seqlen_causal,
                max_seqlen_all,
            )
        case "mps":
            if _mps_kernel_supported(q, k, v):
                return _mps_prefixlm(
                    q,
                    k,
                    v,
                    is_causal,
                    prefix_lens,
                    causal_lens,
                    cu_seqlens,
                    total_seqlen,
                    numseqs,
                    max_seqlen_prefix,
                    max_seqlen_causal,
                    max_seqlen_all,
                )
            return _dense_prefixlm(
                q,
                k,
                v,
                is_causal,
                prefix_lens,
                causal_lens,
                cu_seqlens,
                total_seqlen,
                numseqs,
                max_seqlen_prefix,
                max_seqlen_causal,
                max_seqlen_all,
            )
        case "cpu" | "none":
            return _dense_prefixlm(
                q,
                k,
                v,
                is_causal,
                prefix_lens,
                causal_lens,
                cu_seqlens,
                total_seqlen,
                numseqs,
                max_seqlen_prefix,
                max_seqlen_causal,
                max_seqlen_all,
            )
    raise ValueError(f"Unsupported accelerator_type: {get_accelerator_type()}")
