import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core.frozen_dict import freeze
from typing import Callable


class StdConv(nn.Conv):

    def __std(self, x, axis, eps=1e-5):
        mean = x.mean(axis, keepdims=True)
        var = x.var(axis, keepdims=True)
        return (x - mean) * jnp.sqrt(var + eps)

    def apply(self, params, x):
        params = params.unfreeze()
        params['params']['kernel'] = self.__std(params['params']['kernel'], [0, 1, 2])
        params = freeze(params)
        return super().apply(params, x)


class SinPosEmbs(nn.Module):
    dim : int

    @nn.compact
    def __call__(self, x):
        half_dim = self.dim // 2
        embeddings = jnp.log(10000) / (half_dim - 1)
        embeddings = jnp.exp(- jnp.arange(half_dim) * embeddings)
        embeddings = x[:, None] * embeddings[None, :]
        embeddings = jnp.concatenate((jnp.sin(embeddings), jnp.cos(embeddings)), axis=-1)
        return embeddings


class NetBlock(nn.Module):
    dim: int
    groups: int = 8

    def setup(self):
        self.proj = StdConv(self.dim, (3, 3), padding=1)
        self.norm = nn.GroupNorm(num_groups=self.groups)
        self.act = nn.activation.silu

    def __call__(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    dim: int
    time_emb_dim: tuple = None
    groups: int = 8

    def setup(self):
        self.mlp = nn.Sequential([nn.activation.silu, nn.Dense(self.dim * 2)]) if self.time_emb_dim is not None else None
        self.block1 = NetBlock(self.dim, groups=self.groups)
        self.block2 = NetBlock(self.dim, groups=self.groups)
        self.res_conv = nn.Conv(self.dim, (1, 1))

    def __call__(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = time_emb[:, None, :, :]
            scale_shift = jnp.array_split(time_emb, 2, axis=-1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class PreNorm(nn.Module):
    dim: int
    fn: Callable

    @nn.compact
    def __call__(self, x):
        x = nn.GroupNorm(num_groups=self.dim)(x)
        return self.fn(x)


class LinearAttention(nn.Module):
    dim: int
    heads: int = 4
    dim_head: int = 32

    def setup(self):
        self.scale = self.dim_head ** -0.5
        hidden_dim = self.dim_head * self.heads
        self.to_qkv = nn.Conv(hidden_dim * 3, (1, 1), use_bias=False)
        self.to_out = nn.Sequential([nn.Conv(self.dim, (1, 1)), nn.GroupNorm(num_groups=1)])

    def __call__(self, x):
        b, c, h, w = x.shape
        qkv = jnp.array_split(self.to_qkv(x), 3, axis=-1)
        q, k, v = qkv
        q = nn.activation.softmax(q, axis=-2)
        k = nn.activation.softmax(k, axis=-1)
        q *= self.scale
        context = jnp.einsum("b d n h, b e n h -> b d e h", k, v)
        out = jnp.einsum("b d e h, b d n h -> b e n h", context, q)
        return self.to_out(out)
