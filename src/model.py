import torch.nn as nn
from monai.networks.blocks import UnetOutBlock
from monai.networks.nets import UNet, DynUNet, SwinUNETR as SwinUNETRModel
from monai.bundle import download, load

"""
This module contains the model definitions for 3D UNet, DynUNet, and SwinUNETR.
It also includes functions to load pretrained models from MONAI bundles.
Possible pretrained models:
    - brats_mri_segmentation
    - swin_unetr_btcv_segmentation
    - unet_brats_segmentation
"""

def get_pretrained_bundle_model(bundle_name="brats_mri_segmentation", bundle_dir="models/bundles", out_channels=3):
    print(f"Downloading bundle: {bundle_name}...")
    download(name=bundle_name, bundle_dir=bundle_dir, progress=True)

    print("Loading model from bundle...")
    model = load(
        name=bundle_name,
        bundle_dir=bundle_dir,
        weights_only=False,
        return_state_dict=False
    )

    print("Model loaded successfully.")
    print("Replacing output head to match target classes...")

    # Replace the final output layer (14 → your target classes)
    model.out = UnetOutBlock(
        spatial_dims=3,
        in_channels=model.out.conv.in_channels,
        out_channels=out_channels
    )

    return model


class UNet3D(UNet):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            spatial_dims=3,  # 3D input
            in_channels=in_channels,  # Number of channels in MRI (1 for grayscale)
            out_channels=out_channels,  # Binary segmentation (1 output)
            channels=(16, 32, 64, 128, 256),  # Number of filters in each layer
            strides=(2, 2, 2, 2),  # Strides for downsampling
            num_res_units=2,  # Number of residual units
            dropout=0.3  # Prevents overfitting 
        )
        
class DynUNet(DynUNet):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=[(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)], # Convolution kernel size
            strides=[(1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)], # Strides for downsampling
            upsample_kernel_size=[(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)], # Upsampling kernel size
            norm_name="instance",  # Normalization layer
            deep_supervision=False  # Enable deep supervision
        ) 

class SwinUNETR(SwinUNETRModel):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            use_checkpoint=True,
        )

def get_model(model_type="unet", in_channels=1, out_channels=3, pretrained=False):
    if model_type.lower() == "unet":
        return UNet3D(in_channels=in_channels, out_channels=out_channels)

    elif model_type.lower() == "dynunet":
        return DynUNet(in_channels=in_channels, out_channels=out_channels)

    elif model_type.lower() == "swinunetr":
        if pretrained:
            return get_pretrained_bundle_model("swin_unetr_btcv_segmentation", out_channels=out_channels)
        return SwinUNETR(in_channels=in_channels, out_channels=out_channels)

    else:
        raise ValueError(f"Invalid model type '{model_type}'. Choose from ['unet', 'dynunet', 'swinunetr'].")