import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
from einops import rearrange

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm / (x.size(-1) ** 0.5)
        return self.scale[None, :, None, None] * x / (rms + self.eps)


class SimpleResNetBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding="same", act_fn=torch.sin
    ):
        super(SimpleResNetBlock, self).__init__()
        self.rms_norm = RMSNorm(out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=2)
        self.pointwise = nn.Conv2d(in_channels // 2, out_channels, 1, stride, padding)
        self.act_fn = act_fn
        torch.nn.init.zeros_(self.pointwise.weight)
        torch.nn.init.zeros_(self.pointwise.bias)

    def forward(self, x):
        skip = x
        x = self.rms_norm(x)
        x = self.conv(x)
        lin, gate = rearrange(x, "n (g c) h w -> g n c h w", g=2)
        x = lin * self.act_fn(gate)
        #x = self.act_fn(x)
        x = self.pointwise(x)
        return x + skip
   

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(DownBlock, self).__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(scale_factor)
        self.pointwise_conv = nn.Conv2d(
            in_channels * (scale_factor**2), out_channels, kernel_size=1
        )

    def forward(self, x):
        x = self.pixel_unshuffle(x)
        x = self.pointwise_conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpBlock, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.pointwise_conv = nn.Conv2d(
            in_channels // (scale_factor**2),
            out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.pointwise_conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self, in_channels, out_channels, down_layer_blocks=((32, 2), (64, 2), (128, 2)), act_fn=torch.sin
    ):
        super(Encoder, self).__init__()
        self.in_conv = nn.Conv2d(
            in_channels,
            down_layer_blocks[0][0],
            kernel_size=1,
            stride=1,
            padding="same",
        )

        down_blocks = nn.ModuleList()
        res_blocks = nn.ModuleList()
        for i, blocks in enumerate(down_layer_blocks):
            # (64, 2) dim, block count
            res_block = nn.ModuleList()
            # first down block skipped
            if i != 0:
                downsample = DownBlock(
                    down_layer_blocks[i - 1][0], down_layer_blocks[i][0]
                )
                down_blocks.append(downsample)
            for block in range(blocks[1]):
                # (2) block count
                resnet = SimpleResNetBlock(blocks[0], blocks[0], 3, 1, "same", act_fn)
                res_block.append(resnet)
            res_blocks.append(res_block)

        self.down_blocks = down_blocks
        self.res_blocks = res_blocks

        self.out_norm = RMSNorm(down_layer_blocks[-1][0])
        self.out_conv = nn.Conv2d(
            down_layer_blocks[-1][0],
            out_channels,
            kernel_size=1,
            stride=1,
            padding="same",
        )

    def forward(self, x, checkpoint=True):
        x = self.in_conv(x)
        for i, res_blocks in enumerate(self.res_blocks):
            if i != 0:  # no downscale first input
                x = self.down_blocks[i - 1](x)

            for resnet in res_blocks:
                if checkpoint:
                    x = ckpt.checkpoint(resnet, x)
                else:
                    x = resnet(x)

        x = self.out_norm(x)
        x = self.out_conv(x)
        return F.tanh(x)  # clamp


class Decoder(nn.Module):
    def __init__(
        self, in_channels, out_channels, up_layer_blocks=((128, 2), (64, 2), (32, 2)), act_fn=torch.sin
    ):
        super(Decoder, self).__init__()
        self.in_conv = nn.Conv2d(
            in_channels, up_layer_blocks[0][0], kernel_size=1, stride=1, padding="same"
        )

        up_blocks = nn.ModuleList()
        res_blocks = nn.ModuleList()
        for i, blocks in enumerate(up_layer_blocks):
            # (64, 2) dim, block count
            res_block = nn.ModuleList()
            # first up block skipped
            if i != 0:
                upsample = UpBlock(up_layer_blocks[i - 1][0], up_layer_blocks[i][0])
                up_blocks.append(upsample)
            for block in range(blocks[1]):
                # (2) block count
                resnet = SimpleResNetBlock(blocks[0], blocks[0], 3, 1, "same", act_fn)
                res_block.append(resnet)
            res_blocks.append(res_block)

        self.up_blocks = up_blocks
        self.res_blocks = res_blocks

        self.out_norm = RMSNorm(up_layer_blocks[-1][0])
        self.out_conv = nn.Conv2d(
            up_layer_blocks[-1][0],
            out_channels,
            kernel_size=1,
            stride=1,
            padding="same",
        )

    def forward(self, x, checkpoint=True):
        # checkpointing is defaulted to true
        x = self.in_conv(x)
        for i, res_blocks in enumerate(self.res_blocks):
            if i != 0:  # no downscale first input
                x = self.up_blocks[i - 1](x)

            for resnet in res_blocks:
                if checkpoint:
                    x = ckpt.checkpoint(resnet, x)
                else:
                    x = resnet(x)

        x = self.out_norm(x)
        x = self.out_conv(x)
        return F.tanh(x)  # clamp


class AutoEncoder(nn.Module):
    def __init__(
        self,
        pixel_channels=3,
        bottleneck_channels=4,
        up_layer_blocks=((32, 2), (64, 2), (128, 2)),
        down_layer_blocks=((32, 2), (64, 2), (128, 2)),
        act_fn="sin",
        **kwargs
    ):
        super(AutoEncoder, self).__init__()

        activation_functions = {
            'relu': F.relu,
            'leaky_relu': F.leaky_relu,
            'sigmoid': F.sigmoid,
            'tanh': F.tanh,
            'elu': F.elu,
            'selu': F.selu,
            'gelu': F.gelu,
            'silu': F.silu,
            'sin': torch.sin,
        }

        self.encoder = Encoder(pixel_channels, bottleneck_channels, down_layer_blocks, activation_functions[act_fn])
        self.decoder = Decoder(bottleneck_channels, pixel_channels, up_layer_blocks, activation_functions[act_fn])

    def encode(self, x, checkpoint=True):
        return self.encoder(x, checkpoint)

    def decode(self, x, checkpoint=True):
        return self.decoder(x, checkpoint)

    def forward(self, x):
        x = self.encode(x)
        return self.decode(x)

    def loss_and_grad(
        self,
        x,
        l1=1,
        l2=1,
        checkpoint=True,
        grad_accum_steps=1,
        return_latent=True,
        compute_grad=True,
    ):
        target = x
        latent = None
        x = self.encode(x, checkpoint)
        if return_latent:
            latent = x
        x = self.decode(x, checkpoint)
        # accumulate grad and free the graph
        l1 = torch.mean(torch.abs(x - target)) * l1
        # l1.backward()
        l2 = torch.mean(torch.square(x - target)) * l2
        # l2.backward()
        loss = (l1 + l2) / grad_accum_steps
        if compute_grad:
            loss.backward()
        return loss, l1, l2, latent, x


# Example usage:
if __name__ == "__main__":
    # unit test in here lol :v
    # with torch.no_grad():
    # Dummy input with batch size 1, 3 channels, and 32x32 image
    x = torch.randn(16, 3, 256, 256).to("cuda:0")
    # encoder = Encoder(3, 16)
    # encoder.to("cuda:1")
    # decoder = Decoder(16, 3)
    # decoder.to("cuda:1")
    # latent = encoder(x)
    # output = decoder(latent)
    # print(output.shape)  # Should print torch.Size([1, 16, 32, 32])

    ae = AutoEncoder(
        3,
        4,
        down_layer_blocks=((32, 15), (64, 20), (96, 20), (128, 20)),
        up_layer_blocks=((128, 20), (96, 20), (64, 20), (32, 15)),
    )
    ae.to("cuda:0")
    recon = ae.loss_and_grad(x, checkpoint=True)
    print()
