"""Widening attention heads while keeping the pretrained head inside them.

A pretrained head of width ``n`` embedded in a head of width ``m > n`` computes the same
attention when the added query dimensions and the added output-projection columns start
at zero: the logits ``q . k`` and the output ``proj(P v)`` are then exactly the narrow
head's, while the added key and value dimensions keep their random initialisation, so the
zero slots receive a gradient and the extra width is trainable. Rotary embeddings keep
acting on the native dimensions only, so the pretrained rotation frequencies are untouched.

The attention kernels pad every head to a tile internally (FlashAttention-4 on Hopper: 64
for any narrower head), so up to that width the extra dimensions cost the kernel nothing.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

MIN_HEAD_DIM_KEY = "MIN_HEAD_DIM"
_QUERY = (0,)
_NONE: tuple[int, ...] = ()


@dataclass(frozen=True)
class HeadProjections:
    """The linear layers of an attention module, by their role around the heads.

    A module declares one as its ``head_projections`` class attribute, together with
    ``num_heads``, ``native_head_dim`` and ``head_dim``, and :func:`widen_state_dict`
    embeds checkpoints into it.

    Attributes:
        producing: Layers whose outputs are heads, each with the parts of its weight (0 is
            the query part of a packed qkv) whose added rows start at zero.
        consuming: Layers whose inputs are heads; their added columns start at zero.
    """

    producing: dict[str, tuple[int, ...]]
    consuming: tuple[str, ...]


PACKED_QKV = HeadProjections({"qkv": _QUERY}, ("proj",))
SEPARATE_QKV = HeadProjections({"projq": _QUERY, "projk": _NONE, "projv": _NONE}, ("proj",))


def head_width(channels: int, num_heads: int, min_head_dim: int | None) -> int:
    """The width each head runs at: the native ``channels // num_heads``, or the floor if wider.

    Args:
        channels: Token width the heads are projected from.
        num_heads: Number of heads.
        min_head_dim: Floor on the head width; ``None`` keeps the native width.
    """
    return max(channels // num_heads, min_head_dim or 0)


def widen_state_dict(model: nn.Module, state: dict[str, Tensor]) -> dict[str, Tensor]:
    """A copy of ``state`` with every tensor a widened attention module can absorb embedded
    into that module's shape; every other tensor passes through unchanged.

    Args:
        model: Model the state is about to be loaded into.
        state: Checkpoint tensors keyed like ``model.state_dict()``.
    """
    adapted = dict(state)
    for name, module in model.named_modules():
        if hasattr(module, "head_projections"):
            _widen_module_state(adapted, f"{name}." if name else "", module)
    return adapted


def rotate_first(rope: nn.Module, tokens: Tensor, positions: Tensor, width: int) -> Tensor:
    """``rope`` applied to the first ``width`` dims of every head; the rest pass unrotated.

    Args:
        rope: Rotary module taking ``(tokens, positions)``.
        tokens: ``(..., D)`` with ``D >= width``.
        positions: Positions the module expects.
        width: Dims the pretrained rotation covers.
    """
    if width == tokens.shape[-1]:
        return rope(tokens, positions)
    rotated = rope(tokens[..., :width], positions)
    return torch.cat([rotated, tokens[..., width:]], dim=-1)


def _embed_rows(
    narrow: Tensor, template: Tensor, heads: int, native: int, zero_parts: tuple[int, ...]
) -> Tensor:
    """``narrow``, whose leading dim is ``parts * heads * native``, laid into ``template``'s
    shape: the native rows of every head copied, the new rows of ``zero_parts`` zeroed and
    the other new rows kept from ``template``.

    Args:
        narrow: Checkpoint weight or bias of a head-producing layer.
        template: The wider layer's own tensor, whose values fill the free rows.
        heads: Head count.
        native: The checkpoint's head width.
        zero_parts: Parts whose new rows must start at zero.
    """
    wide = template.detach().clone()
    parts = narrow.shape[0] // (heads * native)
    per_head = template.shape[0] // (parts * heads)
    view = wide.view(parts, heads, per_head, *template.shape[1:])
    view[:, :, :native] = narrow.view(parts, heads, native, *narrow.shape[1:])
    for part in zero_parts:
        view[part, :, native:] = 0
    return wide


def _embed_columns(narrow: Tensor, template: Tensor, heads: int, native: int) -> Tensor:
    """``narrow``, consuming ``native`` dims per head along its last dim, laid into
    ``template``'s shape with zero new columns, so its output is unchanged.

    Args:
        narrow: Checkpoint weight of a head-consuming layer.
        template: The wider layer's own weight, for its shape.
        heads: Head count.
        native: The checkpoint's head width.
    """
    wide = torch.zeros_like(template)
    per_head = template.shape[-1] // heads
    view = wide.view(*template.shape[:-1], heads, per_head)
    view[..., :native] = narrow.view(*narrow.shape[:-1], heads, native)
    return wide


def _embed(
    state: dict[str, Tensor],
    key: str,
    template: Tensor,
    parts: int,
    heads: int,
    native: int,
    zero_parts: tuple[int, ...],
) -> None:
    """Replace ``state[key]`` by its embedding when it has the native width's shape.

    Args:
        state: Checkpoint tensors, edited in place.
        key: Tensor name.
        template: The wider layer's own tensor.
        parts: Parts stacked along its leading dim (three for a packed qkv).
        heads: Head count.
        native: The checkpoint's head width.
        zero_parts: Parts whose new rows must start at zero.
    """
    narrow = state.get(key)
    expected = (parts * heads * native, *template.shape[1:])
    if narrow is not None and tuple(narrow.shape) == expected:
        state[key] = _embed_rows(narrow, template, heads, native, zero_parts)


def _widen_module_state(state: dict[str, Tensor], prefix: str, module: nn.Module) -> None:
    """Rewrite one attention module's projections in ``state`` from the native width.

    A tensor is rewritten only when it has the shape the native width gives it; a matching
    or foreign shape is left for the loader to accept or report.

    Args:
        state: Checkpoint tensors, edited in place.
        prefix: The module's key prefix in ``state``, ending in a dot.
        module: The wider module the tensors are loaded into.
    """
    heads, native, wide = module.num_heads, module.native_head_dim, module.head_dim
    for name, zero_parts in module.head_projections.producing.items():
        layer = getattr(module, name)
        parts = layer.weight.shape[0] // (heads * wide)
        _embed(state, f"{prefix}{name}.weight", layer.weight, parts, heads, native, zero_parts)
        if layer.bias is not None:
            all_parts = tuple(range(parts))
            _embed(state, f"{prefix}{name}.bias", layer.bias, parts, heads, native, all_parts)
    for name in module.head_projections.consuming:
        layer = getattr(module, name)
        key = f"{prefix}{name}.weight"
        narrow = state.get(key)
        expected = (*layer.weight.shape[:-1], heads * native)
        if narrow is not None and tuple(narrow.shape) == expected:
            state[key] = _embed_columns(narrow, layer.weight, heads, native)
