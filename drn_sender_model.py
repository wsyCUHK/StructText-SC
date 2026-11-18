"""Frozen DRN-D-105 loading and sender-side segmentation."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from PIL import Image
MEAN = (0.290101, 0.328081, 0.286964)
STD = (0.182954, 0.186566, 0.184475)
MS_SCALES = (0.5, 0.75, 1.25, 1.5, 1.75)

def load_model(repo_root: Path, checkpoint_path: Path, device_name: str):
    import torch

    source = repo_root / "segment_compat.py"
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing {source}; make a copy of official segment.py with "
            "cuda(async=True) replaced by cuda(non_blocking=True)"
        )
    if not (repo_root / "drn.py").is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing DRN source or checkpoint: {repo_root}, {checkpoint_path}")
    sys.path.insert(0, str(repo_root.resolve()))
    from segment_compat import DRNSeg

    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location="cpu")
    model = DRNSeg("drn_d_105", 19, pretrained_model=None, pretrained=False)
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available()
                          else "cpu" if device_name == "auto" else device_name)
    model = model.to(device).eval()
    print(f"Loaded frozen DRN-D-105 on {device}: {checkpoint_path}", flush=True)
    return model, device

def resize_scores(scores: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Match official DRN test_ms: resize each float32 score plane with PIL bilinear."""
    if scores.shape[1:] == (size[1], size[0]):
        return scores
    return np.stack([
        np.asarray(Image.fromarray(plane).resize(size, Image.Resampling.BILINEAR))
        for plane in scores
    ])

def make_predictor(model, device, multi_scale: bool):
    import torch

    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    def predict(path: Path) -> np.ndarray:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        size = image.size
        scales = (1.0, *MS_SCALES) if multi_scale else (1.0,)
        total = None
        with torch.inference_mode():
            for scale in scales:
                scaled = (max(1, int(size[0] * scale)), max(1, int(size[1] * scale)))
                current = image if scale == 1.0 else image.resize(scaled, Image.Resampling.BICUBIC)
                pixels = np.asarray(current, dtype=np.float32).copy() / 255.0
                batch = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device)
                scores = model((batch - mean) / std)[0][0].cpu().numpy()
                aligned = resize_scores(scores, size)
                total = aligned if total is None else total + aligned
        return total.argmax(axis=0).astype(np.uint8)

    return predict
