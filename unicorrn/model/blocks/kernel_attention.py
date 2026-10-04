import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend
from torch import Tensor
from xformers.ops import memory_efficient_attention

from . import fa4


@nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION)
def flash_attention(q, k, v, **kwargs):
    return F.scaled_dot_product_attention(
        q.bfloat16(), k.bfloat16(), v.bfloat16(), **kwargs
    ).float()


"""
Testing function, do not use
"""


def gaussian_attn(q, k, v, detach=False):
    """
    B, num_seq, H, C_
    """
    # B, num_seq, H, C_ -> B, H, num_seq, C_
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    # B, H, num_q, num_k
    c2 = torch.sum((q.unsqueeze(3) - k.unsqueeze(2)) ** 2, dim=-1)

    # attn = torch.exp(-c2 / eps)
    attn = -c2 / q.shape[-1]
    # print(q @ k.transpose(-2, -1))
    # print(attn)
    attn = F.softmax(attn, dim=-1)
    output = attn @ v
    if detach:
        return output.transpose(1, 2).contiguous(), attn.clone().detach()
    else:
        return output.transpose(1, 2).contiguous(), attn


def gaussian_augment(q, k):
    """Augmented (q', k') whose scaled dot product is -|q - k|^2 / C.

    Args:
        q: Queries ``(B, H, Nq, C)``.
        k: Keys ``(B, H, Nk, C)``.
    """
    q_norm = torch.sum(q ** 2, dim=-1, dtype=q.dtype, keepdim=True)
    k_norm = torch.sum(k ** 2, dim=-1, dtype=k.dtype, keepdim=True)
    q_filler = torch.zeros(*q_norm.shape[:-1], 6, dtype=q.dtype, device=q.device)
    k_filler = torch.zeros(*k_norm.shape[:-1], 6, dtype=k.dtype, device=k.device)
    q_prime = torch.cat([2.0 * q, torch.ones_like(q_norm), -q_norm, q_filler], dim=-1)
    k_prime = torch.cat([k, -k_norm, torch.ones_like(k_norm), k_filler], dim=-1)
    return q_prime, k_prime


def gaussian_memory_efficient_attn(q, k, v, **kwargs):
    q_prime, k_prime = gaussian_augment(q, k)
    attn_out = memory_efficient_attention(
        q_prime, k_prime, v, scale=1 / q.shape[-1], **kwargs
    )

    return attn_out


def gaussian_flash_attn(q, k, v, **kwargs):
    # B, num_seq, H, C_ -> B, H, num_seq, C_
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    v_padding = q.shape[-1] - v.shape[-1] + 8
    v_filler = torch.zeros(*v.shape[:-1], v_padding, dtype=v.dtype, device=v.device)
    q_prime, k_prime = gaussian_augment(q, k)
    attn_out = flash_attention(
        q_prime,
        k_prime,
        torch.cat([v, v_filler], dim=-1),
        scale=1 / q.shape[-1],
        **kwargs,
    )

    return attn_out[..., :-v_padding] if v_padding else attn_out


def gaussian_fa4_attn(q: Tensor, k: Tensor, values: list[Tensor]) -> list[Tensor]:
    """The Gaussian attention of ``gaussian_flash_attn`` for every value stream at once.

    The streams share one attention matrix, and attention is linear in the values, so
    they are concatenated along the head width, attended together and split again --
    exact, and one FlashAttention-4 call per group instead of one per stream. Groups are
    packed up to the augmented query width, so each call keeps equal query and value
    widths, the shape the current kernel runs at. Inputs are fp32 and are cast to bf16 for
    the kernel like ``flash_attention``; the augmentation happens before the cast.

    Args:
        q: Queries ``(B, N, H, C)``.
        k: Keys ``(B, M, H, C)``.
        values: Streams ``(B, M, H, Ci)`` of any widths.

    Returns:
        One ``(B, N, H * Ci)`` fp32 tensor per stream, in the order given.
    """
    q_prime, k_prime = gaussian_augment(q, k)
    q_prime, k_prime = q_prime.bfloat16(), k_prime.bfloat16()
    widths = [v.shape[-1] for v in values]
    outputs: list[Tensor] = []
    for group in _pack_streams(widths, q_prime.shape[-1]):
        packed = torch.cat([values[i].bfloat16() for i in group], dim=-1)
        out = fa4.attention(q_prime, k_prime, packed, softmax_scale=1 / q.shape[-1])
        parts = out.float().split([widths[i] for i in group], dim=-1)
        outputs.extend(part.reshape(*part.shape[:2], -1) for part in parts)
    return outputs


def _split_and_concat_multi_head(tensors, H):
    multi_head_out = []
    dim_per_head = []
    for x in tensors:
        *prefix, D = x.shape
        assert D % H == 0, f"{D} channels not divisible by {H} heads"
        _dim = D // H
        multi_head_out.append(x.view(*prefix, H, _dim))
        dim_per_head.append(_dim)

    return torch.cat(multi_head_out, dim=-1), dim_per_head


def _recover_concat_multi_head(concat_tensor, dim_per_head):
    *prefix, total_dim = concat_tensor.shape

    tensors = []
    offset = 0
    for _dim in dim_per_head:
        part = concat_tensor[..., offset: offset + _dim]
        tensors.append(part.view(*prefix, -1).contiguous())
        offset += _dim

    return tensors


def get_superpoint_mapping(points, mapping=None):
    mapping = (
        torch.arange(points.coord.shape[0], device=points.feat.device)
        if mapping is None
        else mapping
    )
    if "pooling_parent" in points.keys():
        return get_superpoint_mapping(
            points["pooling_parent"], mapping[points["pooling_inverse"]]
        )
    return mapping


def _pack_streams(widths: list[int], limit: int) -> list[list[int]]:
    """Consecutive groups of stream indices whose widths sum to at most ``limit``.

    A stream wider than ``limit`` forms a group of its own.

    Args:
        widths: Head width of every stream, in order.
        limit: Largest total width of one group.
    """
    groups: list[list[int]] = []
    current: list[int] = []
    used = 0
    for index, width in enumerate(widths):
        if current and used + width > limit:
            groups.append(current)
            current, used = [], 0
        current.append(index)
        used += width
    groups.append(current)
    return groups
