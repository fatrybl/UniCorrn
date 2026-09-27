"""FlashAttention-4 (CuTe DSL) behind the fork's attention call sites.

FA4 computes exact attention, so routing a module through it changes no parameter and no
function; only the tensor layout differs from the kernel it replaces. Its one shape rule
on Hopper is that the query and value head widths are multiples of 8 (16 bytes of
fp16/bf16) up to 256, so both wrappers zero-pad the heads up to the next multiple and cut
the padding off the output. Zero query and key columns add nothing to the dot products
and zero value columns produce zero output columns, so the result equals the unpadded
attention to kernel rounding; the softmax scale is taken from the unpadded width.

Install with ``pip install "flash-attn-4[cu13]==4.0.0b32"`` (a pre-release; unpinned needs
``--pre``). The forward runs on Ampere and Hopper, the backward on Hopper (sm90) and newer
only, and there is no attention dropout.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

    FA4_AVAILABLE = True
except ImportError:
    FA4_AVAILABLE = False

HEAD_ALIGNMENT = 8


def configure(module: str, attn_drop: float) -> None:
    """Check at construction that ``module`` can run FlashAttention-4.

    Args:
        module: Name of the module asking for the kernel, for the message.
        attn_drop: Its attention dropout probability, which the kernel cannot apply.

    Raises:
        ImportError: If ``flash_attn.cute`` is not importable.
        ValueError: If attention dropout is requested.
    """
    if not FA4_AVAILABLE:
        raise ImportError(
            f"{module} is configured for FlashAttention-4 but flash_attn.cute is not "
            "importable; install it with  pip install --pre 'flash-attn-4[cu13]'"
        )
    if attn_drop:
        raise ValueError(f"{module}: FlashAttention-4 has no attention dropout, got {attn_drop}")


def attention(q: Tensor, k: Tensor, v: Tensor, softmax_scale: float | None = None) -> Tensor:
    """Dense attention: ``(B, N, H, D)`` queries, ``(B, M, H, D)`` keys, ``(B, M, H, Dv)`` values.

    Args:
        q: Queries, fp16 or bf16.
        k: Keys, same dtype.
        v: Values, same dtype; their head width may differ from the queries'.
        softmax_scale: Logit scale; ``None`` is ``D ** -0.5`` of the unpadded width.

    Returns:
        ``(B, N, H, Dv)`` in the dtype of ``q``.
    """
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    out, _ = flash_attn_func(_pad(q), _pad(k), _pad(v), softmax_scale=scale)
    return out[..., : v.shape[-1]]


def half_attention(q: Tensor, k: Tensor, v: Tensor, softmax_scale: float | None = None) -> Tensor:
    """:func:`attention` for callers in any dtype: fp32 inputs run in bf16, the autocast
    dtype, and the result comes back in the value dtype.

    Args:
        q: Queries.
        k: Keys.
        v: Values.
        softmax_scale: Logit scale; ``None`` is ``D ** -0.5`` of the unpadded width.
    """
    dtype = torch.bfloat16 if v.dtype == torch.float32 else v.dtype
    return attention(q.to(dtype), k.to(dtype), v.to(dtype), softmax_scale).to(v.dtype)


def varlen_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,
    softmax_scale: float | None = None,
) -> Tensor:
    """Attention inside each segment of ``(T, H, D)`` tokens packed along the first dim.

    Args:
        q: Queries, fp16 or bf16.
        k: Keys, packed like the queries.
        v: Values, packed like the queries; their head width may differ.
        cu_seqlens: Int32 segment boundaries, ``(segments + 1,)``, ending at ``T``.
        max_seqlen: Upper bound on a segment's length.
        softmax_scale: Logit scale; ``None`` is ``D ** -0.5`` of the unpadded width.

    Returns:
        ``(T, H, Dv)`` in the dtype of ``q``.
    """
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    out, _ = flash_attn_varlen_func(
        _pad(q),
        _pad(k),
        _pad(v),
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=scale,
    )
    return out[..., : v.shape[-1]]


def _pad(x: Tensor) -> Tensor:
    """``x`` with zero columns appended up to an aligned head width; ``x`` itself if aligned."""
    width = _aligned(x.shape[-1])
    if width == x.shape[-1]:
        return x
    return F.pad(x, (0, width - x.shape[-1]))


def _aligned(width: int) -> int:
    """Smallest multiple of the head alignment that is at least ``width``."""
    return -(-width // HEAD_ALIGNMENT) * HEAD_ALIGNMENT
