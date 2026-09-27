import torch
import torch.nn as nn
from torch import Tensor

from ..embedder import RoPE2D_Continuous, RoPE3D
from .blocks import DropPath, Mlp
from . import fa4
from .kernel_attention import gaussian_fa4_attn, gaussian_flash_attn
from .utils import freeze_modules, offset2batch


def _merge_heads(attn_out, batch, length, channels):
    """Fold ``gaussian_flash_attn``'s (B, H, N, C) output back to (B, N, H*C).

    The kernel returns heads before sequence, unlike the xformers path; reshaping without
    the transpose interleaves the two and is only harmless at a single head.
    """
    return attn_out.transpose(1, 2).reshape(batch, length, channels)


class DualStreamCrossAttentionFA(nn.Module):
    def __init__(
        self, dim, res_dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.num_heads = num_heads

        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj_res = nn.Linear(res_dim, res_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope2d = RoPE2D_Continuous()
        self.rope3d = RoPE3D()

        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def attend(self, q: Tensor, k: Tensor, values: list[Tensor]) -> list[Tensor]:
        """Gaussian attention of every value stream, heads merged into the channels.

        Args:
            q: Queries ``(B, Nq, H, C)``.
            k: Keys ``(B, Nk, H, C)``.
            values: Streams ``(B, Nk, H, Ci)`` sharing the attention matrix.

        Returns:
            One ``(B, Nq, H * Ci)`` tensor per stream.
        """
        return [
            _merge_heads(
                gaussian_flash_attn(q, k, v, dropout_p=self.attn_drop),
                q.shape[0],
                q.shape[1],
                v.shape[2] * v.shape[3],
            )
            for v in values
        ]

    def forward_query_to_img(
        self,
        query,
        key,
        value,
        res,
        qpos,
        kpos,
        img_query,
        appearance_only=False,
        gm_res=None,
    ):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        assert value.shape[:-1] == res.shape[:-1]
        Nv = value.shape[1]
        Cres = res.shape[-1]
        H = self.num_heads

        q = (
            self.projq(query)
            .reshape(B, Nq, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.projk(key)
            .reshape(B, Nk, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        res = res.reshape(B, Nv, self.num_heads, Cres // self.num_heads)

        if not appearance_only:
            if img_query:
                q = self.rope2d(q, qpos)
            else:
                q = self.rope3d(q, qpos)
            k = self.rope2d(k, kpos)

        # (batch_size, seqlen, nheads, headdim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = self.projv(value).reshape(B, Nv, H, C // H)
        # Attention Stream 1 : appearance features
        # Attention Stream 2 : position features
        # (Optional) Attention Stream 3 : GM raw coordinates
        values = [v, res]
        if gm_res is not None:
            values.append(gm_res.reshape(B, Nv, H, 4 // H))
        attn_out = self.attend(q, k, values)
        x = self.proj(attn_out[0])
        x = self.proj_drop(x)
        res_out = self.proj_res(attn_out[1])
        res_out = self.proj_drop(res_out)
        if gm_res is not None:
            return x, res_out, attn_out[2]
        return x, res_out

    def forward_query_to_pcd(
        self,
        query,
        key_batch,
        value_batch,
        res_batch,
        qpos,
        kpos_batch,
        img_query,
        appearance_only=False,
        gm_res_batch=None,
    ):
        B, _, C = query.shape
        Cres = res_batch[0].shape[-1]

        tgt_ = []
        res_ = []
        gm_res_ = []
        for idx in range(B):
            q = query[idx][None]
            k = key_batch[idx]
            v = value_batch[idx]
            res = res_batch[idx]
            kpos = kpos_batch[idx]

            q = (
                self.projq(q)
                .reshape(1, -1, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)
            )
            k = (
                self.projk(k)
                .reshape(1, -1, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)
            )
            res = res.reshape(1, -1, self.num_heads, Cres // self.num_heads)

            if not appearance_only:
                if img_query:
                    q = self.rope2d(q, qpos[idx][None])
                else:
                    q = self.rope3d(q, qpos[idx][None])
                k = self.rope3d(k, kpos[None])
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)

            v = self.projv(v).reshape(1, -1, self.num_heads, C // self.num_heads)
            values = [v, res]
            if gm_res_batch is not None:
                values.append(
                    gm_res_batch[idx].reshape(1, -1, self.num_heads, 4 // self.num_heads)
                )
            attn_out = self.attend(q, k, values)
            tgt_.append(attn_out[0])
            res_.append(attn_out[1])
            if gm_res_batch is not None:
                gm_res_.append(attn_out[2])

        tgt = torch.cat(tgt_, dim=0)
        tgt = self.proj(tgt)
        tgt = self.proj_drop(tgt)
        res = torch.cat(res_, dim=0)
        res = self.proj_res(res)
        res = self.proj_drop(res)

        if gm_res_batch is not None:
            gm_res = torch.cat(gm_res_)
            return tgt, res, gm_res

        return tgt, res


class DualStreamCrossAttentionFA4(DualStreamCrossAttentionFA):
    """``DualStreamCrossAttentionFA`` through FlashAttention-4, the value streams merged
    into as few calls as fit; same parameters, same attention."""

    def __init__(
        self,
        dim: int,
        res_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """Build the parent after checking the kernel can run; ``attn_drop`` must be 0."""
        fa4.configure(type(self).__name__, attn_drop)
        super().__init__(dim, res_dim, num_heads, qkv_bias, attn_drop, proj_drop)

    def attend(self, q: Tensor, k: Tensor, values: list[Tensor]) -> list[Tensor]:
        """All streams through FlashAttention-4 at once; see :func:`gaussian_fa4_attn`."""
        return gaussian_fa4_attn(q, k, values)


class DualStreamQueryDecoderBlockFA(nn.Module):
    attention_cls: type[DualStreamCrossAttentionFA] = DualStreamCrossAttentionFA

    def __init__(
        self,
        dim,
        num_heads,
        res_dim=None,
        mlp_ratio=4,
        qkv_bias=True,
        drop=0.0,
        cross_attn_drop=0.0,
        drop_path=0.0,
        act_layer="gelu",
        norm_layer=nn.LayerNorm,
        norm_mem=True,
        init=False,
        pos_decoder2d=None,
        pos_decoder3d=None,
        **kwargs
    ):
        super().__init__()
        res_dim = dim if res_dim is None else res_dim
        self.cross_attn = self.attention_cls(
            dim,
            res_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=cross_attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm_tgt = norm_layer(dim)
        self.norm_mem = norm_layer(dim) if norm_mem else nn.Identity()
        # self.norm_res = norm_layer(res_dim)

        self.init = init
        if not init:
            assert pos_decoder2d is not None and pos_decoder3d is not None
        self.pos_decoder2d = pos_decoder2d
        self.pos_decoder3d = pos_decoder3d

        self.norm_hidden_ca = norm_layer(res_dim)
        self.norm_hidden_mlp = norm_layer(res_dim)

        self.mlp_hidden = Mlp(
            in_features=res_dim,
            hidden_features=int(res_dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

        self.norm_tgt_ca = norm_layer(dim)
        self.mlp_tgt = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        # self.norm_tgt_mlp = norm_layer(dim)

    def freeze_2d_weights(self):
        freeze_modules(
            self.cross_attn,
            self.drop_path,
            self.norm_tgt,
            self.norm_mem,
            self.norm_hidden_ca,
            self.norm_hidden_mlp,
            self.mlp_hidden,
            self.norm_tgt_ca,
            self.mlp_tgt,
        )

    def forward_query_to_img(
        self, tgt, mem, kpos, res, hidden_state, img_query, gm_res=None
    ):
        tgt = self.norm_tgt(tgt)
        mem = self.norm_mem(mem)
        # res = self.norm_res(res)

        if not self.init:
            if img_query:
                qpos = self.pos_decoder2d(hidden_state)[..., :2]
            else:
                qpos = self.pos_decoder3d(hidden_state)[..., :3]
        else:
            qpos = None
        ret = self.cross_attn.forward_query_to_img(
            query=tgt,
            key=mem,
            value=mem,
            res=res,
            qpos=qpos,
            kpos=kpos,
            img_query=img_query,
            appearance_only=self.init,
            gm_res=gm_res,
        )
        if gm_res is not None:
            tgt2, hidden_tgt, gm_tgt = ret
        else:
            tgt2, hidden_tgt = ret

        # Update
        if not self.init:
            hidden_state = hidden_state + self.drop_path(hidden_tgt)
            hidden_state = self.norm_hidden_ca(hidden_state)
        else:
            hidden_state = hidden_tgt

        hidden_state = hidden_state + self.drop_path(self.mlp_hidden(hidden_state))
        hidden_state = self.norm_hidden_mlp(hidden_state)

        tgt = tgt + self.drop_path(tgt2)
        tgt = self.norm_tgt_ca(tgt)
        tgt = tgt + self.drop_path(self.mlp_tgt(tgt))

        if gm_res is not None:
            return tgt, hidden_state, gm_tgt
        return tgt, hidden_state

    def forward_query_to_pcd(
        self, tgt, mem, kpos, mem_offsets, res, hidden_state, img_query, gm_res=None
    ):
        tgt = self.norm_tgt(tgt)
        mem = self.norm_mem(mem)
        # res = self.norm_res(res)

        mem_batch, kpos_batch = offset2batch(mem, kpos, mem_offsets)
        res_batch = offset2batch(res, kpos, mem_offsets)[0]
        gm_res_batch = (
            offset2batch(gm_res, kpos, mem_offsets)[0] if gm_res is not None else None
        )

        # Cross attention
        if not self.init:
            if img_query:
                qpos = self.pos_decoder2d(hidden_state)[..., :2]
            else:
                qpos = self.pos_decoder3d(hidden_state)[..., :3]
        else:
            qpos = None
        ret = self.cross_attn.forward_query_to_pcd(
            query=tgt,
            key_batch=mem_batch,
            value_batch=mem_batch,
            res_batch=res_batch,
            qpos=qpos,
            kpos_batch=kpos_batch,
            img_query=img_query,
            appearance_only=self.init,
            gm_res_batch=gm_res_batch,
        )
        if gm_res is not None:
            tgt2, hidden_tgt, gm_tgt = ret
        else:
            tgt2, hidden_tgt = ret

        # Update
        if not self.init:
            hidden_state = hidden_state + self.drop_path(hidden_tgt)
            hidden_state = self.norm_hidden_ca(hidden_state)
        else:
            hidden_state = hidden_tgt

        hidden_state = hidden_state + self.drop_path(self.mlp_hidden(hidden_state))
        hidden_state = self.norm_hidden_mlp(hidden_state)

        tgt = tgt + self.drop_path(tgt2)
        tgt = self.norm_tgt_ca(tgt)
        tgt = tgt + self.drop_path(self.mlp_tgt(tgt))

        if gm_res is not None:
            return tgt, hidden_state, gm_tgt
        return tgt, hidden_state


class DualStreamQueryDecoderBlockFA4(DualStreamQueryDecoderBlockFA):
    """The flash-attention decoder block with FlashAttention-4 cross attention."""

    attention_cls = DualStreamCrossAttentionFA4
