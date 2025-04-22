"""
model.py

Factory and custom network definitions for 3D segmentation.
- Implements UNet3D, CustomDynUNet, CustomNNUNet, SegResnet classes tailored for tumor segmentation.
- Provides get_model() to instantiate by name, with optional pretrained bundle support.
- Includes logic to adapt MONAI bundles to single‑channel input and desired output channels.
"""

import logging
import torch.nn as nn
from monai.bundle import download, load
from monai.networks.nets import UNet, DynUNet, SegResNet

logger = logging.getLogger(__name__)


def get_pretrained_bundle_model(
    bundle_name: str = "brats_mri_segmentation",
    bundle_dir: str = "models/bundles",
    out_channels: int = 3,
) -> nn.Module:
    """
    Download and load a pretrained MONAI bundle, adapting it to single-channel input.

    Args:
        bundle_name: Name of the MONAI model bundle.
        bundle_dir: Directory to store/download bundles.
        out_channels: Number of output segmentation channels.

    Returns:
        A PyTorch model with modified first and last conv layers.
    """
    logger.info("Downloading bundle '%s' into %s", bundle_name, bundle_dir)
    download(name=bundle_name, bundle_dir=bundle_dir, progress=True)

    logger.info("Loading model from bundle")
    model = load(
        name=bundle_name,
        bundle_dir=bundle_dir,
        weights_only=False,
        return_state_dict=False,
    )

    # Adjust initial conv for single-channel input
    init_conv = model.convInit.conv
    model.convInit.conv = nn.Conv3d(
        in_channels=1,
        out_channels=init_conv.out_channels,
        kernel_size=init_conv.kernel_size,
        stride=init_conv.stride,
        padding=init_conv.padding,
        bias=init_conv.bias is not None,
    )
    logger.debug("Replaced input conv to accept 1-channel")

    # Adjust final conv to desired out_channels
    final_conv = model.conv_final[2]
    model.conv_final[2] = nn.Conv3d(
        in_channels=final_conv.in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
    )
    logger.debug("Replaced final conv to %d output channels", out_channels)

    return model


class UNet3D(UNet):
    """3D U-Net with residual units and dropout"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: tuple = (32, 64, 128, 256, 400),
        strides: tuple = (2, 2, 2, 2),
        num_res_units: int = 6,
        dropout: float = 0.4,
    ) -> None:
        super().__init__(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            dropout=dropout,
        )


class CustomDynUNet(DynUNet):
    """Customized MONAI DynUNet configuration"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: list = None,
        strides: list = None,
        upsample_kernel_size: list = None,
        norm_name: str = "instance",
        dropout: float = 0.3,
        deep_supervision: bool = False,
    ) -> None:
        kernel_size = kernel_size or [(3, 3, 3)] * 5
        strides = strides or [(1, 1, 1)] + [(2, 2, 2)] * 4
        upsample_kernel_size = upsample_kernel_size or [(2, 2, 2)] * 4
        super().__init__(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=upsample_kernel_size,
            norm_name=norm_name,
            dropout=dropout,
            deep_supervision=deep_supervision,
        )


class CustomNNUNet(UNet):
    """nn-UNet architecture with residual units and dropout"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: tuple = (32, 64, 128, 256, 512),
        strides: tuple = (2, 2, 2, 2),
        num_res_units: int = 2,
        norm: str = "INSTANCE",
        dropout: float = 0.3,
    ) -> None:
        super().__init__(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=norm,
            dropout=dropout,
        )


def get_model(
    model_type: str = "unet",
    in_channels: int = 1,
    out_channels: int = 3,
    pretrained: bool = False,
) -> nn.Module:
    """
    Factory to create segmentation models.

    Args:
        model_type: One of ['unet', 'dynunet', 'nnunet', 'segresnet'].
        in_channels: Number of input channels.
        out_channels: Number of segmentation classes.
        pretrained: Whether to load pretrained weights (for segresnet).

    Returns:
        An instantiated MONAI model.
    """
    mtype = model_type.lower()
    if mtype == "unet":
        return UNet3D(in_channels, out_channels)
    if mtype == "dynunet":
        return CustomDynUNet(in_channels, out_channels)
    if mtype == "nnunet":
        return CustomNNUNet(in_channels, out_channels)
    if mtype == "segresnet":
        if pretrained:
            return get_pretrained_bundle_model(out_channels=out_channels)
        return SegResNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=32,
            dropout_prob=0.4,
            norm_name="instance",
        )

    raise ValueError(f"Invalid model_type '{model_type}'")
