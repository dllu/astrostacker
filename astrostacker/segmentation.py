from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


SAM3_ROOT = Path("/home/dllu/proj/sam3")
SAM3_WEIGHTS_ROOT = Path("/home/dllu/proj/sam3-weights")
BPE_PATH = SAM3_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"


@dataclass(slots=True)
class SegmentationResult:
    sky_mask: np.ndarray
    foreground_mask: np.ndarray
    soft_sky_weight: np.ndarray
    backend: str


def _normalize_segmentation_image(
    image: np.ndarray,
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> np.ndarray:
    data = np.clip(np.asarray(image, dtype=np.float32), 0.0, None)
    lo = float(np.percentile(data, lower_percentile))
    hi = float(np.percentile(data, upper_percentile))
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi) or hi <= lo + 1e-6:
        hi = lo + 1e-6
    normalized = (data - lo) / (hi - lo)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _ensure_sam3_importable() -> bool:
    if not SAM3_ROOT.exists():
        return False
    root = str(SAM3_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return True


def _patch_sam3_fused_addmm() -> bool:
    try:
        import sam3.model.vitdet as vitdet
        import sam3.perflib.fused as fused
    except Exception:
        return False

    if getattr(fused, "_astrostacker_patched", False):
        return True

    def addmm_act_safe(activation, linear, mat1):
        if torch.is_grad_enabled():
            raise ValueError("Expected grad to be disabled.")
        out_dtype = linear.weight.dtype
        work_dtype = torch.bfloat16 if mat1.device.type == "cuda" else out_dtype
        bias = linear.bias.detach().to(work_dtype)
        weight = linear.weight.detach().to(work_dtype)
        mat1_work = mat1.to(work_dtype)
        mat1_flat = mat1_work.view(-1, mat1_work.shape[-1])
        if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
            y = fused.addmm_act_op(
                bias,
                mat1_flat,
                weight.t(),
                beta=1,
                alpha=1,
                use_gelu=False,
            )
        elif activation in [torch.nn.functional.gelu, torch.nn.GELU]:
            y = fused.addmm_act_op(
                bias,
                mat1_flat,
                weight.t(),
                beta=1,
                alpha=1,
                use_gelu=True,
            )
        else:
            raise ValueError(f"Unexpected activation {activation}")
        y = y.view(mat1_work.shape[:-1] + (y.shape[-1],))
        return y.to(out_dtype)

    fused.addmm_act = addmm_act_safe
    vitdet.addmm_act = addmm_act_safe
    fused._astrostacker_patched = True
    return True


def _heuristic_sky_mask(images: list[np.ndarray]) -> np.ndarray:
    resized = np.stack(images, axis=0)
    median = np.median(resized, axis=0)
    gray = cv2.cvtColor(np.clip(median, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.GaussianBlur(np.hypot(grad_x, grad_y), (0, 0), 2.0)
    blue_ratio = median[..., 2] / np.clip(np.mean(median, axis=2), 1e-3, None)

    upper_weight = np.linspace(1.0, 0.3, gray.shape[0], dtype=np.float32)[:, None]
    score = (1.3 - gray) * 0.5 + (blue_ratio - 0.9) * 0.4 + (0.2 - grad) * 2.0
    score = score * upper_weight

    _, binary = cv2.threshold(
        score.astype(np.float32), float(np.median(score)), 1.0, cv2.THRESH_BINARY
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    flood = (binary > 0).astype(np.uint8)
    seed_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8)
    top_connected = np.zeros_like(flood)
    for x in range(0, flood.shape[1], max(flood.shape[1] // 16, 1)):
        if flood[0, x]:
            filled = flood.copy()
            cv2.floodFill(filled, seed_mask, (x, 0), 2)
            top_connected |= filled == 2
            seed_mask.fill(0)

    if not np.any(top_connected):
        top_connected = flood.astype(bool)
    return top_connected.astype(np.uint8)


def _keep_top_connected_region(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary):
        return binary

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary

    top_labels = set(np.unique(labels[0, :]).tolist())
    top_labels.discard(0)

    keep = np.zeros_like(binary)
    min_area = max(64, (binary.shape[0] * binary.shape[1]) // 400)
    if top_labels:
        for label in sorted(top_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= min_area:
                keep[labels == label] = 1
    if not np.any(keep):
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        keep[labels == largest] = 1
    return keep


def _fill_small_holes(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    inverse = 1 - binary
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
    if num_labels <= 1:
        return binary

    filled = binary.copy()
    max_hole_area = max(128, (binary.shape[0] * binary.shape[1]) // 200)
    height, width = binary.shape
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        touches_border = x == 0 or y == 0 or (x + w) >= width or (y + h) >= height
        if not touches_border and area <= max_hole_area:
            filled[labels == label] = 1
    return filled


def _clean_sky_mask(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    binary = _keep_top_connected_region(binary)
    binary = _fill_small_holes(binary)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return binary


def _sam3_masks(
    images: list[np.ndarray], device: str, checkpoint_path: str | None
) -> np.ndarray | None:
    if checkpoint_path is None:
        local_candidates = [*SAM3_WEIGHTS_ROOT.glob("sam3.pt"), *SAM3_ROOT.glob("**/*.pt")]
        checkpoint_path = str(local_candidates[0]) if local_candidates else None
    if checkpoint_path is None:
        return None
    if not _ensure_sam3_importable():
        return None
    if not _patch_sam3_fused_addmm():
        return None
    try:
        from PIL import Image
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except Exception:
        return None

    try:
        model = build_sam3_image_model(
            bpe_path=str(BPE_PATH) if BPE_PATH.exists() else None,
            device=device,
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
        )
        processor = Sam3Processor(model, device=device)
    except Exception:
        return None

    masks = []
    prompts = ["night sky", "sky", "stars in the sky"]
    for image in images:
        state = processor.set_image(Image.fromarray(np.round(image * 255.0).astype(np.uint8)))
        votes = []
        for prompt in prompts:
            output = processor.set_text_prompt(prompt=prompt, state=state)
            prompt_masks = output["masks"]
            if torch.is_tensor(prompt_masks) and prompt_masks.numel() > 0:
                merged = prompt_masks.any(dim=0).squeeze(0).detach().cpu().numpy()
                votes.append(merged.astype(np.uint8))
        if not votes:
            return None
        masks.append(np.sum(votes, axis=0) >= max(1, len(votes) // 2 + 1))
    return np.sum(masks, axis=0) >= max(1, len(masks) // 2 + 1)


def segment_sky(
    previews: list[np.ndarray],
    *,
    dilation_radius: int,
    blur_radius: int,
    sam3_checkpoint: str | None = None,
) -> SegmentationResult:
    normalized_previews = [_normalize_segmentation_image(preview) for preview in previews]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mask = _sam3_masks(normalized_previews, device=device, checkpoint_path=sam3_checkpoint)
    backend = "sam3"
    if mask is None:
        mask = _heuristic_sky_mask(normalized_previews)
        backend = "heuristic"

    mask = _clean_sky_mask(mask)
    fg = 1 - mask
    if dilation_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilation_radius * 2 + 1, dilation_radius * 2 + 1),
        )
        fg = cv2.dilate(fg, kernel)
        mask = 1 - fg

    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), max(blur_radius, 1e-3))
    soft = np.clip(soft, 0.0, 1.0)
    return SegmentationResult(
        sky_mask=mask.astype(bool),
        foreground_mask=fg.astype(bool),
        soft_sky_weight=soft.astype(np.float32),
        backend=backend,
    )
