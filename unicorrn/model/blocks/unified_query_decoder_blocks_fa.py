import torch
import torch.nn as nn

from ..embedder import RoPE2D_Continuous, RoPE3D
from .blocks import DropPath, Mlp
from .kernel_attention import gaussian_flash_attn, gaussian_memory_efficient_attn
from .utils import offset2batch


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

        # Attention Stream 1 : appearance features
        v = self.projv(value).reshape(B, Nv, self.num_heads, C // self.num_heads)
        x = gaussian_flash_attn(q, k, v, dropout_p=self.attn_drop).reshape([B, Nq, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        # Attention Stream 2 : position features
        res_out = gaussian_flash_attn(q, k, res, dropout_p=self.attn_drop).reshape(
            [B, Nq, Cres]
        )
        res_out = self.proj_res(res_out)
        res_out = self.proj_drop(res_out)
        # (Optional) Attention Stream 3 : GM raw coordinates
        if gm_res is not None:
            gm_res = gm_res.reshape(B, Nv, self.num_heads, 4 // self.num_heads)
            gm_out = gaussian_memory_efficient_attn(
                q, k, gm_res, p=self.attn_drop
            ).reshape([B, Nq, 4])
            return x, res_out, gm_out

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
            tgt_.append(
                gaussian_flash_attn(q, k, v, dropout_p=self.attn_drop).reshape(1, -1, C)
            )
            res_.append(
                gaussian_flash_attn(q, k, res, dropout_p=self.attn_drop).reshape(
                    1, -1, Cres
                )
            )

            if gm_res_batch is not None:
                gm_res = gm_res_batch[idx].reshape(
                    1, -1, self.num_heads, 4 // self.num_heads
                )
                gm_res_.append(
                    gaussian_memory_efficient_attn(
                        q, k, gm_res, p=self.attn_drop
                    ).reshape(1, -1, 4)
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
