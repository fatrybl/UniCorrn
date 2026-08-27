import math

import torch
import torch.nn as nn

from ..embedder import RoPE2D_Continuous, RoPE3D
from .blocks import DropPath, Mlp
from .kernel_attention import gaussian_flash_attn, gaussian_logits
from .utils import freeze_modules, offset2batch


_SLOT_BIAS_INIT = -10.0
_SLOT_COORD = 0.5


def attention_statistics(q, k, k_bias, border):
    """Per-query existence cues from the attention distribution, mean over heads:
    ``(slot mass, max image-token probability, normalised entropy, border mass)``.

    Args:
        q: Queries ``(B, Nq, H, C)``.
        k: Keys with the slot last ``(B, Nk + 1, H, C)``.
        k_bias: Key logit biases ``(B, H, Nk + 1)``.
        border: Image tokens on the outer patch ring ``(B, Nk)``.
    """
    prob = torch.softmax(gaussian_logits(q, k, k_bias).float(), dim=-1)
    image = prob[..., :-1]
    entropy = -(prob * torch.log(prob.clamp_min(torch.finfo(prob.dtype).tiny))).sum(-1)
    stats = torch.stack(
        [
            prob[..., -1],
            image.amax(-1),
            entropy / math.log(prob.shape[-1]),
            (image * border[:, None, None, :]).sum(-1),
        ],
        dim=-1,
    )
    return stats.mean(dim=1)


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
        # A learned "no match" key/value: its logit bias starts far below any image token,
        # so the pretrained attention is unchanged until training raises it.
        self.slot_key = nn.Parameter(torch.zeros(dim))
        self.slot_value = nn.Parameter(torch.zeros(dim))
        self.slot_res = nn.Parameter(torch.zeros(res_dim))
        self.slot_bias = nn.Parameter(torch.full((num_heads,), _SLOT_BIAS_INIT))

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
        border=None,
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
        # The slot joins every stream as one more key; its bias rides in the kernel.
        k = torch.cat([k, self.slot_key.view(1, 1, H, C // H).expand(B, 1, H, C // H)], dim=1)
        v = torch.cat([v, self.slot_value.view(1, 1, H, C // H).expand(B, 1, H, C // H)], dim=1)
        res = torch.cat(
            [res, self.slot_res.view(1, 1, H, Cres // H).expand(B, 1, H, Cres // H)], dim=1
        )
        k_bias = torch.cat(
            [torch.zeros(B, H, Nk, device=q.device), self.slot_bias.view(1, H, 1).expand(B, H, 1)],
            dim=-1,
        )
        # Attention Stream 1 : appearance features
        x = _merge_heads(gaussian_flash_attn(q, k, v, k_bias, dropout_p=self.attn_drop), B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # Attention Stream 2 : position features
        res_out = _merge_heads(
            gaussian_flash_attn(q, k, res, k_bias, dropout_p=self.attn_drop), B, Nq, Cres
        )
        res_out = self.proj_res(res_out)
        res_out = self.proj_drop(res_out)
        stats = attention_statistics(q, k, k_bias, border) if border is not None else None
        # (Optional) Attention Stream 3 : GM raw coordinates
        if gm_res is not None:
            slot_gm = gm_res.new_full((B, 1, 4), _SLOT_COORD)
            slot_gm[..., 2:] = 0.0
            gm_res = torch.cat([gm_res, slot_gm], dim=1).reshape(B, Nv + 1, H, 4 // H)
            gm_out = _merge_heads(
                gaussian_flash_attn(q, k, gm_res, k_bias, dropout_p=self.attn_drop), B, Nq, 4
            )
            return x, res_out, gm_out, stats
        return x, res_out, stats

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
            tgt_.append(
                _merge_heads(gaussian_flash_attn(q, k, v, dropout_p=self.attn_drop), 1, -1, C)
            )
            res_.append(
                _merge_heads(
                    gaussian_flash_attn(q, k, res, dropout_p=self.attn_drop), 1, -1, Cres
                )
            )

            if gm_res_batch is not None:
                gm_res = gm_res_batch[idx].reshape(
                    1, -1, self.num_heads, 4 // self.num_heads
                )
                gm_res_.append(
                    _merge_heads(
                        gaussian_flash_attn(q, k, gm_res, dropout_p=self.attn_drop), 1, -1, 4
                    )
                )

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


class DualStreamQueryDecoderBlockFA(nn.Module):
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
        self.cross_attn = DualStreamCrossAttentionFA(
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
        self, tgt, mem, kpos, res, hidden_state, img_query, gm_res=None, border=None
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
            border=border,
        )
        if gm_res is not None:
            tgt2, hidden_tgt, gm_tgt, stats = ret
        else:
            tgt2, hidden_tgt, stats = ret

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
            return tgt, hidden_state, gm_tgt, stats
        return tgt, hidden_state, stats

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
