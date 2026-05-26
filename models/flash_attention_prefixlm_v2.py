from typing import Tuple, Optional

import torch
from torch import Tensor
import numpy as np

from flash_attn_interface import _flash_attn_backward, maybe_contiguous


def compute_aux_seq_tensors_scalars(prefix_lens: np.ndarray, causal_lens: np.ndarray, batch_max_tokens: int):
    # Tensors
    total_lens = prefix_lens + causal_lens
    tensors = {
        "prefix_lens": np.pad(prefix_lens, (0, batch_max_tokens - prefix_lens.shape[0])),
        "causal_lens": np.pad(causal_lens, (0, batch_max_tokens - causal_lens.shape[0])),
        "cu_seqlens": np.pad(np.cumsum(total_lens, dtype=np.int32), (1, batch_max_tokens - total_lens.shape[0] - 1)),
    }
    # Scalars
    scalars = {"total_seqlen": int(total_lens.sum()),
               "numseqs": total_lens.shape[0],
               "max_seqlen_prefix": int(prefix_lens.max()),
               "max_seqlen_causal": int(causal_lens.max()),
               "max_seqlen_all": int(total_lens.max())}
    return tensors, scalars


def _custom_flash_attn_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out_: Optional[Tensor] = None,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_k: Optional[Tensor] = None,
    seqused_q: Optional[Tensor] = None,
    seqused_k: Optional[Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    causal: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Custom implementation of _flash_attn_forward to fix the following issue. Can be removed if the fix is merged to FA3 main branch.
    https://github.com/Dao-AILab/flash-attention/issues/2073"""
    q, k = [maybe_contiguous(x) for x in (q, k)]
    v = v.contiguous() if v.stride(-1) != 1 and v.stride(-3) != 1 else v

    cu_seqlens_q, cu_seqlens_k = [maybe_contiguous(x) for x in (cu_seqlens_q, cu_seqlens_k)]
    seqused_q, seqused_k = [maybe_contiguous(x) for x in (seqused_q, seqused_k)]

    # Call cuda fwd using kwargs for ALL arguments
    out, softmax_lse, *rest = torch.ops.flash_attn_3.fwd(
        q=q,
        k=k,
        v=v,
        k_new=None,
        v_new=None,
        q_v=None,
        out=out_,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cu_seqlens_k_new=None,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        page_table=None,
        kv_batch_idx=None,
        leftpad_k=None,
        rotary_cos=None,
        rotary_sin=None,
        seqlens_rotary=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        softmax_scale=None,
        is_causal=causal,
        window_size_left=-1,
        window_size_right=-1,
        attention_chunk=0,
        softcap=0.0,
        is_rotary_interleaved=True,
        scheduler_metadata=None,
        num_splits=1,
        pack_gqa=None,
        sm_margin=0,
    )
    return out, softmax_lse


@torch.library.custom_op("flash_attn::flash_attn_varlen_prefixlm_compileop", mutates_args=())
def flash_attn_varlen_prefixlm_compileop(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool,
    # CUDA tensors
    prefix_lens: Tensor, causal_lens: Tensor, cu_seqlens: Tensor,
    # CPU tensors (scalars)
    total_seqlen: Tensor, numseqs: Tensor, max_seqlen_prefix: Tensor, max_seqlen_causal: Tensor, max_seqlen_all: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    # Output buffer
    out = torch.empty_like(q)

    # CPU tensor to Python
    assert all(x.device.type == "cpu" for x in (total_seqlen, numseqs, max_seqlen_prefix, max_seqlen_causal, max_seqlen_all))
    total_seqlen_int = total_seqlen.item()
    numseqs_int = numseqs.item()
    max_seqlen_prefix_int: int = max_seqlen_prefix.item()  # pyright: ignore[reportAssignmentType]
    max_seqlen_causal_int: int = max_seqlen_causal.item()  # pyright: ignore[reportAssignmentType]
    max_seqlen_all_int: int = max_seqlen_all.item()  # pyright: ignore[reportAssignmentType]
    # Cut fixed-shape tensors
    cu_seqlens = cu_seqlens[:numseqs_int + 1]
    cu_seqlens_shifted = cu_seqlens + prefix_lens[:numseqs_int + 1]
    prefix_lens = prefix_lens[:numseqs_int]
    causal_lens = causal_lens[:numseqs_int]

    # Fwd pass 1 (bidirectional)
    _, softmax_lse_bidir = _custom_flash_attn_forward(
        out_=out, q=q, k=k, v=v,
        cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        seqused_q=prefix_lens, seqused_k=prefix_lens,
        max_seqlen_q=max_seqlen_prefix_int, max_seqlen_k=max_seqlen_prefix_int,
        causal=is_causal)
    # Fwd pass 2 (causal). If every causal segment is empty, this pass is a
    # no-op; FA3 backward does not reliably handle max_seqlen_q=0.
    if max_seqlen_causal_int > 0:
        _, softmax_lse_causal = _custom_flash_attn_forward(
            out_=out, q=q, k=k, v=v,
            cu_seqlens_q=cu_seqlens_shifted, cu_seqlens_k=cu_seqlens,
            seqused_q=causal_lens,
            max_seqlen_q=max_seqlen_causal_int, max_seqlen_k=max_seqlen_all_int,
            causal=True)
    else:
        softmax_lse_causal = torch.empty_like(softmax_lse_bidir)

    out[total_seqlen_int:] = 0  # Zero padding
    return out, softmax_lse_bidir, softmax_lse_causal


@torch.library.register_fake("flash_attn::flash_attn_varlen_prefixlm_compileop")
def fake_flash_attn_varlen_prefixlm_compileop(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool,
    # CUDA tensors
    prefix_lens: Tensor, causal_lens: Tensor, cu_seqlens: Tensor,
    # CPU tensors (scalars)
    total_seqlen: Tensor, numseqs: Tensor, max_seqlen_prefix: Tensor, max_seqlen_causal: Tensor, max_seqlen_all: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    out = torch.empty_like(q)
    softmax_lse_bidir = torch.empty((q.shape[-2], q.shape[-3]), dtype=torch.float32, device=q.device)  # type: ignore
    softmax_lse_causal = torch.empty((q.shape[-2], q.shape[-3]), dtype=torch.float32, device=q.device)  # type: ignore

    return out, softmax_lse_bidir, softmax_lse_causal


@torch.library.custom_op("flash_attn::flash_attn_varlen_prefixlm_bwd_compileop", mutates_args=())
def flash_attn_varlen_prefixlm_bwd_compileop(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool,
    dout: Tensor,

    out: Tensor,
    softmax_lse_bidir: Tensor,
    softmax_lse_causal: Tensor,
    # CUDA tensors
    prefix_lens: Tensor, causal_lens: Tensor, cu_seqlens: Tensor,
    # CPU tensors (scalars)
    total_seqlen: Tensor, numseqs: Tensor, max_seqlen_prefix: Tensor, max_seqlen_causal: Tensor, max_seqlen_all: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    # CPU tensor to Python
    assert all(x.device.type == "cpu" for x in (total_seqlen, numseqs, max_seqlen_prefix, max_seqlen_causal, max_seqlen_all))
    total_seqlen_int = total_seqlen.item()
    numseqs_int = numseqs.item()
    max_seqlen_prefix_int = max_seqlen_prefix.item()
    max_seqlen_causal_int = max_seqlen_causal.item()
    max_seqlen_all_int = max_seqlen_all.item()
    # Cut fixed-shape tensors
    cu_seqlens = cu_seqlens[:numseqs_int + 1]
    cu_seqlens_shifted = cu_seqlens + prefix_lens[:numseqs_int + 1]
    prefix_lens = prefix_lens[:numseqs_int]
    causal_lens = causal_lens[:numseqs_int]

    # Buffers
    dq = torch.empty_like(q)
    dk1, dv1 = torch.zeros_like(k), torch.zeros_like(v)  # Zero-fill in advance
    dk2, dv2 = torch.zeros_like(k), torch.zeros_like(v)
    # Bwd pass 1 (bidirectional)
    _flash_attn_backward(
        dout=dout, q=q, k=k, v=v, out=out,
        softmax_lse=softmax_lse_bidir,
        cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen_prefix_int, max_seqlen_k=max_seqlen_prefix_int,
        sequed_q=prefix_lens, sequed_k=prefix_lens,
        dq=dq,
        dk=dk1,
        dv=dv1,
        is_causal=is_causal)
    # Bwd pass 2 (causal)
    if max_seqlen_causal_int > 0:
        _flash_attn_backward(
            dout=dout, q=q, k=k, v=v, out=out,
            softmax_lse=softmax_lse_causal,
            cu_seqlens_q=cu_seqlens_shifted, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen_causal_int, max_seqlen_k=max_seqlen_all_int,
            sequed_q=causal_lens,
            dq=dq,
            dk=dk2,
            dv=dv2,
            is_causal=True)

    # Zero padding grads
    dq[total_seqlen_int:] = 0
    dk2[total_seqlen_int:] = 0
    dv2[total_seqlen_int:] = 0
    return dq, dk1.add_(dk2), dv1.add_(dv2)


@torch.library.register_fake("flash_attn::flash_attn_varlen_prefixlm_bwd_compileop")
def fake_flash_attn_varlen_prefixlm_bwd_compileop(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool,
    dout: Tensor,

    out: Tensor,
    softmax_lse_bidir: Tensor,
    softmax_lse_causal: Tensor,
    # CUDA tensors
    prefix_lens: Tensor, causal_lens: Tensor, cu_seqlens: Tensor,
    # CPU tensors (scalars)
    total_seqlen: Tensor, numseqs: Tensor, max_seqlen_prefix: Tensor, max_seqlen_causal: Tensor, max_seqlen_all: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


class FlashAttnVarlenPrefixLM(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                q: Tensor,
                k: Tensor,
                v: Tensor,
                is_causal: bool,
                # CUDA tensors
                prefix_lens: Tensor, causal_lens: Tensor, cu_seqlens: Tensor,
                # CPU tensors (scalars)
                total_seqlen: Tensor, numseqs: Tensor, max_seqlen_prefix: Tensor, max_seqlen_causal: Tensor, max_seqlen_all: Tensor):
        out, softmax_lse_bidir, softmax_lse_causal = flash_attn_varlen_prefixlm_compileop(q, k, v, is_causal,
                                                                                          prefix_lens, causal_lens, cu_seqlens,
                                                                                          total_seqlen, numseqs, max_seqlen_prefix, max_seqlen_causal, max_seqlen_all)
        ctx.save_for_backward(q, k, v, out, softmax_lse_bidir, softmax_lse_causal,
                              prefix_lens, causal_lens, cu_seqlens,
                              total_seqlen, numseqs, max_seqlen_prefix, max_seqlen_causal, max_seqlen_all)
        ctx.is_causal = is_causal
        return out

    @staticmethod
    def backward(ctx, dout: Tensor):  # pyright: ignore[reportIncompatibleMethodOverride]
        q, k, v, out, softmax_lse_bidir, softmax_lse_causal, \
            prefix_lens, causal_lens, cu_seqlens, \
            total_seqlen, numseqs, max_seqlen_prefix, max_seqlen_causal, max_seqlen_all = ctx.saved_tensors

        dq, dk, dv = flash_attn_varlen_prefixlm_bwd_compileop(
            q, k, v, ctx.is_causal,
            dout,
            out,
            softmax_lse_bidir, softmax_lse_causal,
            prefix_lens, causal_lens, cu_seqlens,
            total_seqlen, numseqs, max_seqlen_prefix, max_seqlen_causal, max_seqlen_all)

        return dq, dk, dv, *((None, ) * 9)


def flash_attn_varlen_prefixlm(q: Tensor,
                               k: Tensor,
                               v: Tensor,
                               is_causal: bool,  # Compat for causal attention. Use same prefixLM passes but less efficient.
                               # CUDA tensors
                               prefix_lens: Tensor, causal_lens: Tensor, cu_seqlens: Tensor,
                               # CPU tensors (scalars)
                               total_seqlen: Tensor, numseqs: Tensor, max_seqlen_prefix: Tensor, max_seqlen_causal: Tensor, max_seqlen_all: Tensor):
    # Apply function
    return FlashAttnVarlenPrefixLM.apply(q, k, v, is_causal,
                                         prefix_lens, causal_lens, cu_seqlens,
                                         total_seqlen, numseqs, max_seqlen_prefix, max_seqlen_causal, max_seqlen_all)
