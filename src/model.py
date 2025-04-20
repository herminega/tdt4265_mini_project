import torch.nn as nn
from monai.networks.nets import UNet, DynUNet, UNet as NNUNet, SegResNet
from monai.bundle import download, load

"""
This module contains the model definitions for 3D UNet, DynUNet, and SwinUNETR.
It also includes functions to load pretrained models from MONAI bundles.
Possible pretrained models:
    - brats_mri_segmentation
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

    # Fix input channels (BraTS model has 4 → yours is 1)
    old_conv = model.convInit.conv
    model.convInit.conv = nn.Conv3d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )
    print("Replaced `convInit` to accept 1-channel input.")

    # Fix output channels (BraTS has 3 classes → yours is 3 too, but just in case)
    model.conv_final[2] = nn.Conv3d(
        in_channels=model.conv_final[2].in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=1
    )
    print("Replaced final output layer to match", out_channels, "classes.")

    return model


class UNet3D(UNet):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            spatial_dims=3,  # 3D input
            in_channels=in_channels,  # Number of channels in MRI (1 for grayscale)
            out_channels=out_channels,  # Binary segmentation (1 output)
            channels=(32, 64, 128, 256, 400),  # Number of channels in each layer
            strides=(2, 2, 2, 2),  # Strides for downsampling
            num_res_units=6,  # Number of residual units
            dropout=0.4  # Prevents overfitting 
        )

class DynUNet(DynUNet):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=[(3, 3, 3)] * 5, # Convolution kernel size
            strides=[(1, 1, 1)] + [(2, 2, 2)] * 4, # Strides for downsampling
            upsample_kernel_size=[(2, 2, 2)] * 4, # Upsampling kernel size
            norm_name="instance",  # Normalization layer
            dropout=0.3,  # Dropout to prevent overfitting
            deep_supervision=False  # Enable deep supervision
        ) 


class NNUNet(NNUNet):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2, 2),
            num_res_units=2,
            norm="INSTANCE",
            dropout=0.3,
        )        
   
def get_model(model_type="unet", in_channels=1, out_channels=3, pretrained=False):
    if model_type.lower() == "unet":
        return UNet3D(in_channels=in_channels, out_channels=out_channels)

    elif model_type.lower() == "dynunet":
        return DynUNet(in_channels=in_channels, out_channels=out_channels)
    
    elif model_type.lower() == "nnunet":
        return NNUNet(in_channels=in_channels, out_channels=out_channels)

    elif model_type.lower() == "segresnet":
        if pretrained:
            # If you have a pretrained bundle for SegResNet, for example "segresnet_brats_segmentation",
            # you can load it using the get_pretrained_bundle_model function:
            return get_pretrained_bundle_model("brats_mri_segmentation", out_channels=out_channels)
        else:
            # Otherwise, initialize SegResNet with a typical configuration.
            return SegResNet(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=out_channels,
                init_features=32,  # you might adjust this based on memory or expected capacity
                dropout_prob=0.4,  # Dropout to prevent overfitting
                norm_name="instance",  # Normalization layer
            )

    else:
        raise ValueError(f"Invalid model type '{model_type}'. Choose from ['unet', 'dynunet', 'nnunet', 'swinunetr', 'segresnet'].")
 
    