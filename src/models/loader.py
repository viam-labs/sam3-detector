"""Load SAM3 image and video models onto the selected device.

The upstream `Sam3VideoPredictor` always calls `.cuda()`, which breaks CPU/MPS
fallback. We wrap `Sam3BasePredictor` and `build_sam3_video_model` so the same
GPU detection path as sam2-detector (cuda / mps / cpu) works here.
"""

from __future__ import annotations

import inspect
from typing import Any, Optional

from viam.logging import getLogger

from models.common import (
    resolve_checkpoint_path,
    select_device,
    wrap_checkpoint_error,
)

LOGGER = getLogger(__name__)


def load_image_processor(device: str, confidence_threshold: float = 0.5):
    """Build a Sam3Processor on `device` using a bundled or HF checkpoint."""
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    checkpoint_path = resolve_checkpoint_path()
    LOGGER.debug(
        f"Loading SAM3 image model (checkpoint={checkpoint_path or 'huggingface'}) "
        f"on {device}"
    )
    try:
        model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=checkpoint_path is None,
            device=device,
            eval_mode=True,
            enable_segmentation=True,
        )
    except Exception as err:
        raise wrap_checkpoint_error(err) from err
    return Sam3Processor(
        model, device=device, confidence_threshold=confidence_threshold
    )


class DeviceAwareSam3VideoPredictor:
    """Video predictor that honors cuda / mps / cpu instead of forcing CUDA.

    Request API matches the official predictor: `handle_request` /
    `handle_stream_request` with start_session, add_prompt, propagate_in_video,
    reset_session, close_session.
    """

    def __init__(self, device: str):
        from sam3.model.sam3_base_predictor import Sam3BasePredictor
        from sam3.model_builder import build_sam3_video_model

        checkpoint_path = resolve_checkpoint_path()
        LOGGER.debug(
            f"Loading SAM3 video model (checkpoint={checkpoint_path or 'huggingface'}) "
            f"on {device}"
        )
        self.device = device
        self._base = Sam3BasePredictor()
        self._base.async_loading_frames = False
        self._base.video_loader_type = "cv2"
        try:
            self._base.model = build_sam3_video_model(
                checkpoint_path=checkpoint_path,
                load_from_HF=checkpoint_path is None,
                device=device,
            ).eval()
        except Exception as err:
            raise wrap_checkpoint_error(err) from err
        # Expose session bookkeeping used by Sam3BasePredictor helpers.
        self._all_inference_states = self._base._all_inference_states

    def handle_request(self, request: dict) -> dict:
        request_type = request.get("type")
        if request_type == "add_prompt":
            return self._add_prompt(request)
        return self._base.handle_request(request)

    def handle_stream_request(self, request: dict):
        yield from self._base.handle_stream_request(request)

    def _add_prompt(self, request: dict) -> dict:
        """Same as Sam3BasePredictor.add_prompt, with device-aware autocast."""
        import torch

        session = self._base._get_session(request["session_id"])
        inference_state = session["state"]
        self._base._extend_expiration_time(session)

        text = request.get("text", None)
        points = request.get("points", None)
        point_labels = request.get("point_labels", None)
        bounding_boxes = request.get("bounding_boxes", None)
        bounding_box_labels = request.get("bounding_box_labels", None)

        if points is not None and not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if point_labels is not None and not isinstance(point_labels, torch.Tensor):
            point_labels = torch.tensor(point_labels, dtype=torch.int32)
        if bounding_boxes is not None and not isinstance(bounding_boxes, torch.Tensor):
            bounding_boxes = torch.tensor(bounding_boxes, dtype=torch.float32)
        if bounding_box_labels is not None and not isinstance(
            bounding_box_labels, torch.Tensor
        ):
            bounding_box_labels = torch.tensor(bounding_box_labels, dtype=torch.int32)

        kwargs = dict(
            inference_state=inference_state,
            frame_idx=request["frame_index"],
            text_str=text,
            points=points,
            point_labels=point_labels,
            clear_old_points=request.get("clear_old_points", True),
            boxes_xywh=bounding_boxes,
            box_labels=bounding_box_labels,
            clear_old_boxes=request.get("clear_old_boxes", True),
            output_prob_thresh=request.get("output_prob_thresh", 0.5),
            rel_coordinates=request.get("rel_coordinates", True),
        )
        if request.get("obj_id") is not None:
            kwargs["obj_id"] = request["obj_id"]

        sig = inspect.signature(self._base.model.add_prompt)
        valid_params = set(sig.parameters.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        autocast_device = "cuda" if self.device == "cuda" else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        with torch.inference_mode():
            if self.device in ("cuda", "cpu"):
                with torch.autocast(device_type=autocast_device, dtype=dtype):
                    frame_idx, outputs = self._base.model.add_prompt(**filtered_kwargs)
            else:
                frame_idx, outputs = self._base.model.add_prompt(**filtered_kwargs)
        return {"frame_index": frame_idx, "outputs": outputs}


def load_video_predictor(device: str) -> Optional[Any]:
    """Load the video predictor. Returns None on non-CUDA devices when the
    official video I/O path would call `.cuda()` internally despite our
    wrapper (some helpers still hardcode CUDA). We still try on cuda/mps/cpu
    via DeviceAwareSam3VideoPredictor; callers should fall back to the image
    processor if this raises.
    """
    return DeviceAwareSam3VideoPredictor(device=device)


def video_predictor_supported(device: str) -> bool:
    """SAM3's video I/O still contains `.cuda()` calls unless frames are
    offloaded. We enable the video path on CUDA; image-processor fallback
    everywhere else (including MPS).
    """
    return device == "cuda"


# Re-export so callers can `from models.loader import select_device`.
__all__ = [
    "load_image_processor",
    "load_video_predictor",
    "video_predictor_supported",
    "select_device",
]
