import flax.linen as nn
from .blocks import ResnetBlock, PreNorm, LinearAttention, SinPosEmbs


class Unet(nn.Module):
    dims = (32, 64)
    time_dim = 4
    channels = 1
    self_condition = False
    resnet_block_groups = 4

    def setup(self):
        input_channels = self.channels * (2 if self.self_condition else 1)
        self.init_conv = nn.Conv(input_channels, (1, 1))
        dim_in, out_dim = self.dims

        time_dim = self.time_dim * 4
        self.time_mlp = nn.Sequential([
            SinPosEmbs(self.time_dim),
            nn.Dense(time_dim),
            nn.activation.gelu,
            nn.Dense(time_dim),
        ])
        self.block1 = nn.Sequential([
            ResnetBlock(dim_in, time_emb_dim=time_dim),
            ResnetBlock(dim_in, time_emb_dim=time_dim),
            ResnetBlock(dim_in, time_emb_dim=time_dim)
        ])
        self.attn1 = PreNorm(dim_in, LinearAttention(dim_in))
        self.block2 = nn.Sequential([
            ResnetBlock(out_dim, time_emb_dim=time_dim),
            ResnetBlock(out_dim, time_emb_dim=time_dim),
            ResnetBlock(out_dim, time_emb_dim=time_dim)
        ])
        self.attn2 = PreNorm(out_dim, LinearAttention(out_dim))
        self.final = nn.Conv(input_channels, (1, 1))

    def __call__(self, x, time):
        x = self.init_conv(x)
        time = self.time_mlp(time)
        x = self.block1(x, time)
        x = self.attn1(x)
        x = self.block2(x, time)
        x = self.attn2(x)
        x = self.final(x)
        return x
