"""Convert SAM3 processor / video-predictor outputs into detection records.

Kept free of Viam and SAM3 imports so unit tests can run without those
packages. The vision services wrap `RawDetection` in `viam.proto...Detection`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class RawDetection:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    confidence: float
    class_name: str
    object_id: Optional[int] = None


def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Convert a binary mask (H, W) to (x_min, y_min, x_max, y_max)."""
    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
        if mask.ndim > 2:
            mask = mask[0]
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def as_numpy(value: Any) -> Optional[np.ndarray]:
    """Detach / convert tensors and array-likes to ndarray. None stays None."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _clip_box(x_min: int, y_min: int, x_max: int, y_max: int) -> Optional[Tuple[int, int, int, int]]:
    if x_max <= x_min or y_max <= y_min:
        return None
    return x_min, y_min, x_max, y_max


def detections_from_xyxy(
    boxes: Any,
    scores: Any,
    label: str,
    min_score: float = 0.0,
) -> List[RawDetection]:
    """SAM3 image processor boxes are pixel xyxy; scores are probabilities."""
    boxes_np = as_numpy(boxes)
    scores_np = as_numpy(scores)
    if boxes_np is None or boxes_np.size == 0:
        return []
    boxes_np = np.atleast_2d(boxes_np)
    if scores_np is None:
        scores_np = np.ones(len(boxes_np), dtype=np.float32)
    else:
        scores_np = np.asarray(scores_np).reshape(-1)

    out: List[RawDetection] = []
    for i, box in enumerate(boxes_np):
        score = float(scores_np[i]) if i < len(scores_np) else 1.0
        if score < min_score:
            continue
        x0, y0, x1, y1 = (float(v) for v in box[:4])
        clipped = _clip_box(int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))
        if clipped is None:
            continue
        x_min, y_min, x_max, y_max = clipped
        out.append(
            RawDetection(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                confidence=score,
                class_name=label,
                object_id=i,
            )
        )
    return out


def detections_from_masks(
    masks: Any,
    scores: Any,
    label: str,
    min_score: float = 0.0,
) -> List[RawDetection]:
    """Fall back to axis-aligned boxes around SAM3 binary masks."""
    masks_np = as_numpy(masks)
    scores_np = as_numpy(scores)
    if masks_np is None or masks_np.size == 0:
        return []
    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]
    if scores_np is None:
        scores_np = np.ones(len(masks_np), dtype=np.float32)
    else:
        scores_np = np.asarray(scores_np).reshape(-1)

    out: List[RawDetection] = []
    for i, mask in enumerate(masks_np):
        score = float(scores_np[i]) if i < len(scores_np) else 1.0
        if score < min_score:
            continue
        bbox = mask_to_bbox(mask)
        if bbox is None:
            continue
        x_min, y_min, x_max, y_max = bbox
        out.append(
            RawDetection(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                confidence=score,
                class_name=label,
                object_id=i,
            )
        )
    return out


def rel_xywh_to_xyxy(
    box: Sequence[float],
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    """Convert a box to pixel xyxy.

    SAM3 video outputs use relative xywh in [0, 1]. Absolute pixel xywh is
    accepted when any component is clearly larger than 1.5.
    """
    x, y, w, h = (float(v) for v in box[:4])
    if max(abs(x), abs(y), abs(w), abs(h)) <= 1.5:
        x0 = x * width
        y0 = y * height
        x1 = (x + w) * width
        y1 = (y + h) * height
        return x0, y0, x1, y1
    return x, y, x + w, y + h


def detections_from_video_outputs(
    outputs: dict,
    width: int,
    height: int,
    label: str,
    min_score: float = 0.0,
) -> List[RawDetection]:
    """Map SAM3 video-predictor frame outputs to detections.

    Expected keys (any may be missing):
      out_boxes_xywh, out_probs, out_obj_ids, out_binary_masks
    """
    boxes = as_numpy(outputs.get("out_boxes_xywh"))
    probs = as_numpy(outputs.get("out_probs"))
    obj_ids = as_numpy(outputs.get("out_obj_ids"))
    masks = as_numpy(outputs.get("out_binary_masks"))

    n = 0
    for candidate in (obj_ids, boxes, masks, probs):
        if candidate is not None and getattr(candidate, "shape", None) is not None:
            n = max(n, len(candidate))
    if n == 0:
        return []

    out: List[RawDetection] = []
    for i in range(n):
        score = float(probs[i]) if probs is not None and i < len(probs) else 1.0
        if score < min_score:
            continue
        obj_id = int(obj_ids[i]) if obj_ids is not None and i < len(obj_ids) else i

        bbox: Optional[Tuple[int, int, int, int]] = None
        if boxes is not None and i < len(boxes):
            x0, y0, x1, y1 = rel_xywh_to_xyxy(boxes[i], width, height)
            bbox = _clip_box(int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))
        if bbox is None and masks is not None and i < len(masks):
            bbox = mask_to_bbox(masks[i])
        if bbox is None:
            continue
        x_min, y_min, x_max, y_max = bbox
        out.append(
            RawDetection(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                confidence=score,
                class_name=label,
                object_id=obj_id,
            )
        )
    return out


def detections_from_processor_state(
    state: dict,
    label: str,
    min_score: float = 0.0,
) -> List[RawDetection]:
    """Read `boxes` / `scores` / `masks` from a Sam3Processor state dict."""
    dets = detections_from_xyxy(state.get("boxes"), state.get("scores"), label, min_score)
    if dets:
        return dets
    return detections_from_masks(state.get("masks"), state.get("scores"), label, min_score)


def to_iterable(dets: Iterable[RawDetection]) -> List[RawDetection]:
    return list(dets)
