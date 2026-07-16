#!/usr/bin/env python3
"""Super-resolution architectures reproduced from xinntao/Real-ESRGAN and XPixelGroup/BasicSR.

Reproduced rather than depending on the `realesrgan`/`basicsr` packages directly, since these
are small, self-contained architectures and pulling in those full packages (and their own
dependency trees) just for two model classes isn't worth the weight. Verified this session by
loading each official checkpoint with `strict=True` (every layer name/shape must match exactly)
and by visually comparing real output against the official implementation.

Weight initialization (`default_init_weights` in the original source) is omitted: it only seeds
random values before training, and every model here always has a pretrained state dict loaded
immediately after construction, which overwrites it completely -- so it has no effect on the
converted output either way.
"""

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812 (torch's own idiomatic import alias)


class SRVGGNetCompact(nn.Module):
    """A compact VGG-style super-resolution network. Reproduced from `realesrgan/archs/srvgg_arch.py`.

    Used by `realesr-animevideov3` and `realesr-general-x4v3`.
    """

    def __init__(  # noqa: PLR0913
        self, num_in_ch: int = 3, num_out_ch: int = 3, num_feat: int = 64,
        num_conv: int = 16, upscale: int = 4, act_type: str = 'prelu',
    ) -> None:
        """Build the conv stack for the given hyperparameters (see the checkpoint's own docs).

        Args:
            num_in_ch (int, optional): Channel number of inputs. Defaults to 3.
            num_out_ch (int, optional): Channel number of outputs. Defaults to 3.
            num_feat (int, optional): Channel number of intermediate features. Defaults to 64.
            num_conv (int, optional): Number of convolution layers in the body network.
                Defaults to 16.
            upscale (int, optional): Upsampling factor. Defaults to 4.
            act_type (str, optional): Activation type -- `"relu"`, `"prelu"`, or `"leakyrelu"`.
                Defaults to `"prelu"`.
        """
        super().__init__()
        self.upscale = upscale

        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(self._activation(act_type, num_feat))

        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(self._activation(act_type, num_feat))

        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    @staticmethod
    def _activation(act_type: str, num_feat: int) -> nn.Module:
        """Build the activation layer named by `act_type`.

        Args:
            act_type (str): Activation type -- `"relu"`, `"prelu"`, or `"leakyrelu"`.
            num_feat (int): Channel number of intermediate features (only used by `"prelu"`,
                which learns one slope per channel).

        Returns:
            nn.Module: The constructed activation layer.

        Raises:
            NotImplementedError: If `act_type` isn't one of the three supported values.
        """
        if act_type == 'relu':
            return nn.ReLU(inplace=True)
        if act_type == 'prelu':
            return nn.PReLU(num_parameters=num_feat)
        if act_type == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)
        raise NotImplementedError(f'activation {act_type!r} not implemented')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the compact conv stack, pixel-shuffle it up, and add the residual.

        Args:
            x (torch.Tensor): Input image batch, shape `(batch, num_in_ch, height, width)`.

        Returns:
            torch.Tensor: Upscaled image batch, shape
            `(batch, num_out_ch, height * upscale, width * upscale)`.
        """
        out = x
        for layer in self.body:
            out = layer(out)

        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        out += base
        return out


class _ResidualDenseBlock(nn.Module):
    """Residual Dense Block, used inside `_RRDB`. Reproduced from `basicsr/archs/rrdbnet_arch.py`."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        """Build the five densely-connected conv layers.

        Args:
            num_feat (int, optional): Channel number of intermediate features. Defaults to 64.
            num_grow_ch (int, optional): Channels added by each dense layer. Defaults to 32.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Densely-connected conv stack; the 0.2 residual scale matches the official source.

        Args:
            x (torch.Tensor): Input features, shape `(batch, num_feat, height, width)`.

        Returns:
            torch.Tensor: Same shape as `x`.
        """
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class _RRDB(nn.Module):
    """Residual in Residual Dense Block, used inside `RRDBNet`."""

    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        """Build the three stacked `_ResidualDenseBlock`s.

        Args:
            num_feat (int): Channel number of intermediate features.
            num_grow_ch (int, optional): Channels added by each dense layer. Defaults to 32.
        """
        super().__init__()
        self.rdb1 = _ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = _ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = _ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Stack of three dense blocks; the 0.2 residual scale matches the official source.

        Args:
            x (torch.Tensor): Input features, shape `(batch, num_feat, height, width)`.

        Returns:
            torch.Tensor: Same shape as `x`.
        """
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """ESRGAN-style network used by `RealESRGAN_x4plus`/`RealESRGAN_x4plus_anime_6B`.

    Only the `scale=4` path is reproduced (matching every model this tool currently converts) --
    the official architecture's `scale=1`/`scale=2` pixel-unshuffle preprocessing is not
    included.
    """

    def __init__(  # noqa: PLR0913
        self, num_in_ch: int = 3, num_out_ch: int = 3, scale: int = 4,
        num_feat: int = 64, num_block: int = 23, num_grow_ch: int = 32,
    ) -> None:
        """Build the trunk (`num_block` RRDB blocks) and upsampling head for `scale=4`.

        Args:
            num_in_ch (int, optional): Channel number of inputs. Defaults to 3.
            num_out_ch (int, optional): Channel number of outputs. Defaults to 3.
            scale (int, optional): Upsampling factor. Only `4` is reproduced/verified.
                Defaults to 4.
            num_feat (int, optional): Channel number of intermediate features. Defaults to 64.
            num_block (int, optional): Number of RRDB blocks in the trunk. Defaults to 23.
            num_grow_ch (int, optional): Channels added by each dense layer. Defaults to 32.

        Raises:
            NotImplementedError: If `scale` isn't `4`.
        """
        super().__init__()
        if scale != 4:  # noqa: PLR2004
            raise NotImplementedError('only scale=4 is reproduced/verified by this tool')

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[_RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Trunk + residual, then two nearest-neighbor 2x upsampling stages.

        Args:
            x (torch.Tensor): Input image batch, shape `(batch, num_in_ch, height, width)`.

        Returns:
            torch.Tensor: Upscaled image batch, shape
            `(batch, num_out_ch, height * 4, width * 4)`.
        """
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))
