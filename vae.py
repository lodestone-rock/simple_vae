import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_init(layer):
    # set initial params to zero
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


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
        self, in_channels, out_channels, kernel_size=3, stride=1, padding="same"
    ):
        super(SimpleResNetBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.rms_norm = RMSNorm(out_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, stride, padding)
        torch.nn.init.zeros_(self.pointwise.weight)
        torch.nn.init.zeros_(self.pointwise.bias)

    def forward(self, x):
        skip = x
        x = self.conv(x)
        x = self.rms_norm(x)
        x = torch.sin(x)
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
        self, in_channels, out_channels, down_layer_blocks=((32, 2), (64, 2), (128, 2))
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
                resnet = SimpleResNetBlock(blocks[0], blocks[0], 3, 1, "same")
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

    def forward(self, x):
        x = self.in_conv(x)
        for i, res_blocks in enumerate(self.res_blocks):
            if i != 0:  # no downscale first input
                x = self.down_blocks[i - 1](x)
            for resnet in res_blocks:
                x = resnet(x)
        x = self.out_norm(x)
        x = self.out_conv(x)
        return F.tanh(x)  # clamp


class Decoder(nn.Module):
    def __init__(
        self, in_channels, out_channels, up_layer_blocks=((32, 2), (64, 2), (128, 2))
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
                resnet = SimpleResNetBlock(blocks[0], blocks[0], 3, 1, "same")
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

    def forward(self, x):
        x = self.in_conv(x)
        for i, res_blocks in enumerate(self.res_blocks):
            if i != 0:  # no downscale first input
                x = self.up_blocks[i - 1](x)
            for resnet in res_blocks:
                x = resnet(x)
        x = self.out_norm(x)
        x = self.out_conv(x)
        return x


# Example usage:
if __name__ == "__main__":
    with torch.no_grad():
        # Dummy input with batch size 1, 3 channels, and 32x32 image
        x = torch.randn(24, 3, 256, 256).to("cuda:1")
        encoder = Encoder(3, 16)
        encoder.to("cuda:1")
        decoder = Decoder(16, 3)
        decoder.to("cuda:1")
        latent = encoder(x)
        output = decoder(latent)
        print(output.shape)  # Should print torch.Size([1, 16, 32, 32])
