"""Cityscapes palette and SwinJSCC model configuration."""
from __future__ import annotations
import argparse
from types import SimpleNamespace
import numpy as np

PALETTE = np.array(
    [
        (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
        (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
        (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
        (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100),
        (0, 0, 230), (119, 11, 32),
    ],
    dtype=np.uint8,
)

MODEL = "SwinJSCC_w/o_SAandRA"

def nearest_train_ids(rgb: np.ndarray, chunk_pixels: int = 65536) -> np.ndarray:
    """Quantize continuous SwinJSCC RGB output to the 19 evaluated colours."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected H x W x 3 uint8 RGB image")
    flat = rgb.reshape(-1, 3)
    result = np.empty(flat.shape[0], dtype=np.uint8)
    palette = PALETTE.astype(np.int32)
    for start in range(0, flat.shape[0], chunk_pixels):
        stop = min(start + chunk_pixels, flat.shape[0])
        delta = flat[start:stop].astype(np.int32)[:, None, :] - palette[None, :, :]
        squared_distance = np.sum(delta * delta, axis=2)
        result[start:stop] = np.argmin(squared_distance, axis=1).astype(np.uint8)
    return result.reshape(rgb.shape[:2])

def model_config(torch, args: argparse.Namespace) -> tuple[SimpleNamespace, SimpleNamespace]:
    import torch.nn as nn

    depths = {
        "small": ([2, 2, 2, 2], [2, 2, 2, 2]),
        "base": ([2, 2, 6, 2], [2, 6, 2, 2]),
    }
    encoder_depths, decoder_depths = depths[args.model_size]
    common = dict(model=MODEL, img_size=(args.crop_size, args.crop_size),
                  C=args.C, window_size=8, mlp_ratio=4., qkv_bias=True,
                  qk_scale=None, norm_layer=nn.LayerNorm, patch_norm=True)
    encoder_kwargs = dict(common, patch_size=2, in_chans=3,
                          embed_dims=[128, 192, 256, 320],
                          depths=encoder_depths, num_heads=[4, 6, 8, 10])
    decoder_kwargs = dict(common, embed_dims=[320, 256, 192, 128],
                          depths=decoder_depths, num_heads=[10, 8, 6, 4])
    network_args = SimpleNamespace(trainset="Cityscapes", model=MODEL,
                                   C=str(args.C), multiple_snr=str(args.snr),
                                   channel_type="awgn", distortion_metric="MSE")
    config = SimpleNamespace(logger=None, pass_channel=True, CUDA=True,
                             device=torch.device("cuda:0"), norm=False,
                             downsample=4, encoder_kwargs=encoder_kwargs,
                             decoder_kwargs=decoder_kwargs)
    return network_args, config
