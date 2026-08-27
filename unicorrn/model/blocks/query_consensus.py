"""Self-attention among 3D queries with rotary relative positions.

A query's readout can then draw on its spatial neighbours: the frustum is a contiguous
region bounded by planes, so neighbouring points share membership. The output projection
starts at zero, so the block is the identity until trained. Positions are unit-sphere
coordinates scaled so the rotary bands resolve neighbourhoods rather than the whole scene.
"""

import torch
from torch import nn
from torch.nn import functional as F

from ..embedder import RoPE3D

_POS_SCALE = 64.0


class QueryConsensus(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.rope3d = RoPE3D()

    def forward(self, x, pos):
        batch, length, dim = x.shape
        heads = self.num_heads
        q, k, v = (
            self.qkv(self.norm(x))
            .reshape(batch, length, 3, heads, dim // heads)
            .permute(2, 0, 3, 1, 4)
        )
        scaled = pos * _POS_SCALE
        q, k = self.rope3d(q, scaled).to(v.dtype), self.rope3d(k, scaled).to(v.dtype)
        out = F.scaled_dot_product_attention(q, k, v)
        return x + self.proj(out.transpose(1, 2).reshape(batch, length, dim))
