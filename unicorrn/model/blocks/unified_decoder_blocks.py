import torch
import torch.nn as nn
from torch import Tensor
from xformers.ops import memory_efficient_attention

from ..embedder import RoPE2D_Continuous, RoPE3D
from . import fa4
from .attention_kernels import FA4, XFORMERS, check_kernel
from .blocks import DropPath, Mlp
from .head_widening import PACKED_QKV, SEPARATE_QKV, head_width, rotate_first
from .point_transformer_v3 import offset2bincount
from .utils import batch2offset, offset2batch


def _attention(q: Tensor, k: Tensor, v: Tensor, p: float, kernel: str, scale: float) -> Tensor:
    """Attention with q/k cast to the value dtype (autocast-safe), through ``kernel``."""
    if kernel == FA4:
        return fa4.half_attention(q, k, v, scale)
    return memory_efficient_attention(q.to(v.dtype), k.to(v.dtype), v, p=p, scale=scale)


class MMEfficientAttention(nn.Module):
    """
    Multi-Modal (MM) Efficient Self Attention block

    """

    head_projections = PACKED_QKV

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        pcd_patch_size=1024,
        order_index=0,
        attn_kernel: str = XFORMERS,
        min_head_dim: int | None = None,
    ):
        super().__init__()
        check_kernel(attn_kernel, (XFORMERS, FA4), type(self).__name__, attn_drop)
        self.attn_kernel = attn_kernel
        self.num_heads = num_heads
        self.native_head_dim = dim // num_heads
        self.head_dim = head_width(dim, num_heads, min_head_dim)
        self.scale = self.native_head_dim**-0.5
        inner = num_heads * self.head_dim
        self.qkv = nn.Linear(dim, inner * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(inner, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.patch_size = 0
        self.patch_size_max = pcd_patch_size
        self.order_index = order_index
        self.rope2d = RoPE2D_Continuous()
        self.rope3d = RoPE3D()

    @torch.no_grad()
    def get_pos(self, point, order):
        # Voxel positions: the scale the released fusion weights were trained with.
        pos_key = f"pos_grid_{self.order_index}_{self.patch_size}"
        if pos_key not in point.keys():
            point[pos_key] = point.grid_coord[order]
        return point[pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = f"pad_{self.patch_size}"
        unpad_key = f"unpad_{self.patch_size}"
        cu_seqlens_key = f"cu_seqlens_{self.patch_size}"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )

        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward_img_tokens(self, x, xpos):
        B, N, _ = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .transpose(1, 3)
        )
        q, k, v = [qkv[:, :, i] for i in range(3)]  # B x num_heads x N x head_dim

        q = rotate_first(self.rope2d, q, xpos, self.native_head_dim)
        k = rotate_first(self.rope2d, k, xpos, self.native_head_dim)

        # (batch_size, seqlen, nheads, headdim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        x = _attention(q, k, v, self.attn_drop, self.attn_kernel, self.scale)
        x = x.reshape([B, N, self.num_heads * self.head_dim])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_pcd_tokens(self, x, point):
        self.patch_size = min(
            offset2bincount(point.offset).min().tolist(), self.patch_size_max
        )

        H = self.num_heads
        K = self.patch_size
        D = self.head_dim

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(x)[order]

        q, k, v = qkv.reshape(-1, K, 3, H, D).permute(2, 0, 3, 1, 4).unbind(dim=0)

        pos = self.get_pos(point, order).reshape(-1, K, 3)

        # apply Rotary Position Embedding to the native part of every head
        q = rotate_first(self.rope3d, q, pos, self.native_head_dim)
        k = rotate_first(self.rope3d, k, pos, self.native_head_dim)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        feat = _attention(q, k, v, self.attn_drop, self.attn_kernel, self.scale)
        feat = feat.reshape(-1, H * D)

        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        return feat


class MMEfficientCrossAttention(nn.Module):
    """
    Multi-Modal (MM) Efficient Cross Attention block

    """

    head_projections = SEPARATE_QKV

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_kernel: str = XFORMERS,
        min_head_dim: int | None = None,
    ):
        super().__init__()
        check_kernel(attn_kernel, (XFORMERS, FA4), type(self).__name__, attn_drop)
        self.attn_kernel = attn_kernel
        self.num_heads = num_heads
        self.native_head_dim = dim // num_heads
        self.head_dim = head_width(dim, num_heads, min_head_dim)
        self.scale = self.native_head_dim**-0.5
        inner = num_heads * self.head_dim

        self.projq = nn.Linear(dim, inner, bias=qkv_bias)
        self.projk = nn.Linear(dim, inner, bias=qkv_bias)
        self.projv = nn.Linear(dim, inner, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(inner, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope2d = RoPE2D_Continuous()
        self.rope3d = RoPE3D()

    def forward_img_to_img(self, query, key, value, qpos, kpos):
        B, Nq, _ = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]

        q = self.projq(query).reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.projk(key).reshape(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.projv(value).reshape(B, Nv, self.num_heads, self.head_dim)

        q = rotate_first(self.rope2d, q, qpos, self.native_head_dim)
        k = rotate_first(self.rope2d, k, kpos, self.native_head_dim)

        # (batch_size, seqlen, nheads, headdim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)

        x = _attention(q, k, v, self.attn_drop, self.attn_kernel, self.scale)
        x = x.reshape([B, Nq, self.num_heads * self.head_dim])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_img_to_pcd(self, query, key, value, qpos, kpos, tgt_point_offset):
        B, Nq, _ = query.shape

        # image
        q = self.projq(query).reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = rotate_first(self.rope2d, q, qpos, self.native_head_dim)

        key_batch, kpos_batch = offset2batch(key, kpos, tgt_point_offset)
        val_batch, _ = offset2batch(value, kpos, tgt_point_offset)

        x = []
        for i in range(B):
            q_i, k_i, v_i = q[i][None], key_batch[i], val_batch[i]

            Nk = key_batch[i].shape[0]  # number of points x dim
            Nv = val_batch[i].shape[0]  # number of points x dim

            # point
            k_i = self.projk(k_i).reshape(1, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v_i = self.projv(v_i).reshape(1, Nv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            k_i = rotate_first(self.rope3d, k_i, kpos_batch[i][None], self.native_head_dim)

            # (batch_size, seqlen, nheads, headdim)
            q_i = q_i.permute(0, 2, 1, 3)
            k_i = k_i.permute(0, 2, 1, 3)
            v_i = v_i.permute(0, 2, 1, 3)

            x.append(_attention(q_i, k_i, v_i, self.attn_drop, self.attn_kernel, self.scale))

        x = torch.vstack(x)
        x = x.reshape([B, Nq, self.num_heads * self.head_dim])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_pcd_to_img(self, query, key, value, qpos, kpos, src_point_offset):
        B, Nk, _ = key.shape
        Nv = value.shape[1]

        k = self.projk(key).reshape(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.projv(value).reshape(B, Nv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        k = rotate_first(self.rope2d, k, kpos, self.native_head_dim)

        query_batch, qpos_batch = offset2batch(query, qpos, src_point_offset)

        x = []
        for i in range(B):
            q_i, k_i, v_i = query_batch[i], k[i][None], v[i][None]

            Nq = q_i.shape[0]

            q_i = self.projq(q_i).reshape(1, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            q_i = rotate_first(self.rope3d, q_i, qpos_batch[i][None], self.native_head_dim)

            q_i = q_i.permute(0, 2, 1, 3)
            k_i = k_i.permute(0, 2, 1, 3)
            v_i = v_i.permute(0, 2, 1, 3)

            output = _attention(q_i, k_i, v_i, self.attn_drop, self.attn_kernel, self.scale)
            x.append(output.reshape(1, Nq, self.num_heads * self.head_dim))

        x = batch2offset(x)  # (Npoints, inner)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_pcd_to_pcd(
        self, query, key, value, qpos, kpos, src_point_offset, tgt_point_offset
    ):
        assert tgt_point_offset.shape[0] == src_point_offset.shape[0]
        query_batch, qpos_batch = offset2batch(query, qpos, src_point_offset)
        key_batch, kpos_batch = offset2batch(key, kpos, tgt_point_offset)
        val_batch, _ = offset2batch(value, kpos, tgt_point_offset)

        x = []
        for i in range(len(query_batch)):
            q_i, k_i, v_i = query_batch[i], key_batch[i], val_batch[i]

            Nq = q_i.shape[0]
            Nk = k_i.shape[0]
            Nv = v_i.shape[0]

            q_i = self.projq(q_i).reshape(1, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            q_i = rotate_first(self.rope3d, q_i, qpos_batch[i][None], self.native_head_dim)
            k_i = self.projk(k_i).reshape(1, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k_i = rotate_first(self.rope3d, k_i, kpos_batch[i][None], self.native_head_dim)
            v_i = self.projv(v_i).reshape(1, Nv, self.num_heads, self.head_dim)

            q_i = q_i.permute(0, 2, 1, 3)
            k_i = k_i.permute(0, 2, 1, 3)

            output = _attention(q_i, k_i, v_i, self.attn_drop, self.attn_kernel, self.scale)
            x.append(output.reshape(1, Nq, self.num_heads * self.head_dim))

        x = batch2offset(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MMDecoderBlock(nn.Module):
    """
    Multi-Modal (MM) Decoder block
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer="gelu",
        norm_layer=nn.LayerNorm,
        norm_mem=True,
        order_index=0,
        pcd_patch_size=1024,
        attn_kernel: str = XFORMERS,
        min_head_dim: int | None = None,
    ):
        super().__init__()

        self.self_attn = MMEfficientAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            pcd_patch_size=pcd_patch_size,
            order_index=order_index,
            attn_kernel=attn_kernel,
            min_head_dim=min_head_dim,
        )
        self.cross_attn = MMEfficientCrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_kernel=attn_kernel,
            min_head_dim=min_head_dim,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.norm_tgt = norm_layer(dim) if norm_mem else nn.Identity()
        self.order_index = order_index

    def forward_img_to_img(self, src, tgt, src_pos, tgt_pos):
        src = src + self.drop_path(
            self.self_attn.forward_img_tokens(self.norm1(src), src_pos)
        )
        tgt = self.norm_tgt(tgt)
        src = src + self.drop_path(
            self.cross_attn.forward_img_to_img(
                self.norm2(src), tgt, tgt, src_pos, tgt_pos
            )
        )
        src = src + self.drop_path(self.mlp(self.norm3(src)))
        return src

    def forward_img_to_pcd(self, src, tgt, src_pos, tgt_point):
        order = tgt_point.serialized_order[self.order_index]
        tgt = tgt[order]
        tgt_pos = tgt_point.coord[order]

        src = src + self.drop_path(
            self.self_attn.forward_img_tokens(self.norm1(src), src_pos)
        )
        tgt = self.norm_tgt(tgt)
        src = src + self.drop_path(
            self.cross_attn.forward_img_to_pcd(
                self.norm2(src), tgt, tgt, src_pos, tgt_pos, tgt_point.offset
            )
        )
        src = src + self.drop_path(self.mlp(self.norm3(src)))
        return src

    def forward_pcd_to_img(self, src, tgt, src_point, tgt_pos):
        src = src + self.drop_path(
            self.self_attn.forward_pcd_tokens(self.norm1(src), src_point)
        )

        order = src_point.serialized_order[self.order_index]
        inverse = src_point.serialized_inverse[self.order_index]
        src = src[order]
        src_pos = src_point.coord[order]

        tgt = self.norm_tgt(tgt)
        src = src[inverse] + self.drop_path(
            self.cross_attn.forward_pcd_to_img(
                self.norm2(src), tgt, tgt, src_pos, tgt_pos, src_point.offset
            )[inverse]
        )
        src = src + self.drop_path(self.mlp(self.norm3(src)))
        return src

    def forward_pcd_to_pcd(self, src, tgt, src_point, tgt_point):
        src_order = src_point.serialized_order[self.order_index]
        tgt_order = tgt_point.serialized_order[self.order_index]
        src_inverse = src_point.serialized_inverse[self.order_index]
        src = src[src_order]
        tgt = tgt[tgt_order]
        src_pos = src_point.coord[src_order]
        tgt_pos = tgt_point.coord[tgt_order]

        src = src + self.drop_path(
            self.self_attn.forward_pcd_tokens(self.norm1(src), src_point)
        )
        tgt = self.norm_tgt(tgt)
        src = src[src_inverse] + self.drop_path(
            self.cross_attn.forward_pcd_to_pcd(
                self.norm2(src),
                tgt,
                tgt,
                src_pos,
                tgt_pos,
                src_point.offset,
                tgt_point.offset,
            )[src_inverse]
        )
        src = src + self.drop_path(self.mlp(self.norm3(src)))
        return src


class MMDecoderBlockBidirectional(nn.Module):
    """
    Multi-Modal (MM) Bidectional Decoder block
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer="gelu",
        norm_layer=nn.LayerNorm,
        norm_mem=True,
        order_index=0,
        pcd_patch_size=1024,
        attn_kernel: str = XFORMERS,
        min_head_dim: int | None = None,
    ):
        super().__init__()
        self.decoder_block = MMDecoderBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            norm_mem=norm_mem,
            order_index=order_index,
            pcd_patch_size=pcd_patch_size,
            attn_kernel=attn_kernel,
            min_head_dim=min_head_dim,
        )

    def forward_img_to_img(self, src, tgt, src_pos, tgt_pos):
        src_ = self.decoder_block.forward_img_to_img(src, tgt, src_pos, tgt_pos)
        tgt_ = self.decoder_block.forward_img_to_img(tgt, src, tgt_pos, src_pos)
        return src_, tgt_

    def forward_img_to_pcd(self, src, tgt, src_pos, tgt_point):
        src_ = self.decoder_block.forward_img_to_pcd(src, tgt, src_pos, tgt_point)
        tgt_ = self.decoder_block.forward_pcd_to_img(tgt, src, tgt_point, src_pos)
        return src_, tgt_

    def forward_pcd_to_img(self, src, tgt, src_point, tgt_pos):
        src_ = self.decoder_block.forward_pcd_to_img(src, tgt, src_point, tgt_pos)
        tgt_ = self.decoder_block.forward_img_to_pcd(tgt, src, tgt_pos, src_point)
        return src_, tgt_

    def forward_pcd_to_pcd(self, src, tgt, src_point, tgt_point):
        src_ = self.decoder_block.forward_pcd_to_pcd(src, tgt, src_point, tgt_point)
        tgt_ = self.decoder_block.forward_pcd_to_pcd(tgt, src, tgt_point, src_point)
        return src_, tgt_

