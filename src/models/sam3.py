"""
SAM3 detector: open-vocabulary text prompts, no bounding box required.

Unlike SAM2 (which needed an initial click / box to start tracking), SAM3
detects every instance of a short noun phrase such as "stemless wine glass"
from the label alone.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from typing import ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

# Must be set before torch is imported anywhere — required for AMD ROCm GPUs.
if os.path.exists("/opt/rocm") and "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import models.tqdm_silence  # noqa: F401  — silence tqdm before SAM3 imports it

import numpy as np
from PIL import Image as PILImage
from typing_extensions import Self
from viam.components.camera import Camera
from viam.logging import getLogger
from viam.media.video import ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import Classification, Detection, GetPropertiesResponse
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.vision import CaptureAllResult, Vision
from viam.utils import ValueTypes

from models.boxes import RawDetection, detections_from_processor_state, detections_from_video_outputs
from models.common import (
    SAM3_MODEL_ID,
    numpy_to_pil,
    sam3_amp_context,
    select_device,
    torch_build_info,
    viam_image_to_numpy,
)
from models.loader import load_image_processor, load_video_predictor, video_predictor_supported

LOGGER = getLogger(__name__)

DEFAULT_MAX_FRAMES = 300
DEFAULT_CONFIDENCE = 0.5


def _to_viam(det: RawDetection) -> Detection:
    return Detection(
        x_min=det.x_min,
        y_min=det.y_min,
        x_max=det.x_max,
        y_max=det.y_max,
        confidence=det.confidence,
        class_name=det.class_name,
    )


class Sam3(Vision, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam", "sam3-detector"), "sam3")

    _device: str = "cpu"
    _label: str = "object"
    _confidence_threshold: float = DEFAULT_CONFIDENCE
    _max_frames: int = DEFAULT_MAX_FRAMES
    _camera_name: str = ""
    _camera: ResourceBase
    _initial_point: Optional[Tuple[int, int]] = None
    _use_video: bool = False
    _processor: object = None
    _video_predictor: object = None

    _frame_dir: Optional[str] = None
    _frame_count: int = 0
    _window_start: int = 0
    _frames_since_propagation: int = 0
    _propagation_interval: int = 1
    _last_frame_size: Tuple[int, int] = (0, 0)  # (width, height)

    _detections: Dict[int, List[Detection]] = {}
    _last_detections: List[Detection]
    _lock: threading.Lock

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        instance = super().new(config, dependencies)
        instance._lock = threading.Lock()
        instance._detections = {}
        instance._last_detections = []
        instance._frame_count = 0
        instance._window_start = 0
        instance._frames_since_propagation = 0
        attrs = config.attributes.fields

        instance._camera_name = attrs["camera_name"].string_value
        instance._camera = dependencies[Camera.get_resource_name(instance._camera_name)]
        LOGGER.debug(f"Using camera: {instance._camera_name}")

        instance._label = attrs["label"].string_value
        LOGGER.info(f"Text prompt (label): {instance._label!r}")

        if "confidence_threshold" in attrs:
            instance._confidence_threshold = attrs["confidence_threshold"].number_value
        if "propagation_interval" in attrs:
            instance._propagation_interval = int(attrs["propagation_interval"].number_value)
        if "max_frames" in attrs:
            instance._max_frames = int(attrs["max_frames"].number_value)
        else:
            instance._max_frames = DEFAULT_MAX_FRAMES

        if "initial_point_x" in attrs and "initial_point_y" in attrs:
            instance._initial_point = (
                int(attrs["initial_point_x"].number_value),
                int(attrs["initial_point_y"].number_value),
            )
            LOGGER.debug(f"Optional point prompt: {instance._initial_point}")
        else:
            instance._initial_point = None

        instance._frame_dir = tempfile.mkdtemp(prefix="sam3_frames_")
        instance._device = select_device()
        instance._use_video = video_predictor_supported(instance._device)

        # Image processor is always loaded: it is the CPU/MPS path and the
        # per-frame fallback if video propagation fails.
        instance._processor = load_image_processor(
            instance._device, confidence_threshold=instance._confidence_threshold
        )
        instance._video_predictor = None
        if instance._use_video:
            try:
                instance._video_predictor = load_video_predictor(instance._device)
                LOGGER.info("SAM3 video predictor loaded (text-prompt tracking)")
            except Exception as err:
                LOGGER.warning(
                    f"SAM3 video predictor failed to load ({err}); "
                    f"falling back to per-frame image detection"
                )
                instance._use_video = False
                instance._video_predictor = None
        else:
            LOGGER.info(
                f"SAM3 per-frame image detector loaded on {instance._device} "
                f"(video tracking requires CUDA)"
            )
        return instance

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = config.attributes.fields
        if "camera_name" not in attrs or not attrs["camera_name"].string_value:
            raise ValueError("camera_name is required")
        if "label" not in attrs or not attrs["label"].string_value:
            raise ValueError(
                "label is required — SAM3 uses it as an open-vocabulary text "
                "prompt (e.g. 'stemless wine glass'). No bounding box is needed."
            )
        has_x = "initial_point_x" in attrs
        has_y = "initial_point_y" in attrs
        if has_x != has_y:
            raise ValueError(
                "Must provide both initial_point_x and initial_point_y, or neither"
            )
        return [attrs["camera_name"].string_value], []

    async def _get_camera_image(self) -> np.ndarray:
        images, _ = await self._camera.get_images()
        return viam_image_to_numpy(images[0])

    def _save_frame(self, image_np: np.ndarray) -> int:
        local_idx = self._frame_count - self._window_start
        path = os.path.join(self._frame_dir, f"{local_idx}.jpg")
        PILImage.fromarray(image_np).save(path, quality=90)
        self._last_frame_size = (image_np.shape[1], image_np.shape[0])
        frame_idx = self._frame_count
        self._frame_count += 1
        self._frames_since_propagation += 1
        window_size = self._frame_count - self._window_start
        if window_size > self._max_frames:
            self._compact_window()
        return frame_idx

    def _compact_window(self):
        window_size = self._frame_count - self._window_start
        if window_size <= self._max_frames:
            return
        keep_count = self._max_frames
        drop_count = window_size - keep_count
        new_start = self._window_start + drop_count
        new_dir = tempfile.mkdtemp(prefix="sam3_frames_")
        for new_idx in range(keep_count):
            old_idx = drop_count + new_idx
            old_path = os.path.join(self._frame_dir, f"{old_idx}.jpg")
            new_path = os.path.join(new_dir, f"{new_idx}.jpg")
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
        shutil.rmtree(self._frame_dir, ignore_errors=True)
        self._frame_dir = new_dir
        self._window_start = new_start
        LOGGER.debug(f"Compacted window: dropped {drop_count} frames, keeping {keep_count}")

    def _run_image_on_array(self, image_np: np.ndarray) -> List[Detection]:
        pil = numpy_to_pil(image_np)
        with sam3_amp_context(self._device):
            state = self._processor.set_image(pil)
            state = self._processor.set_text_prompt(self._label, state)
        raw = detections_from_processor_state(
            state, self._label, min_score=self._confidence_threshold
        )
        return [_to_viam(d) for d in raw]

    def _run_image_propagation(self, image_np: np.ndarray, frame_idx: int) -> List[Detection]:
        dets = self._run_image_on_array(image_np)
        self._detections[frame_idx] = dets
        self._last_detections = dets
        self._frames_since_propagation = 0
        return dets

    def _run_video_propagation(self):
        window_size = self._frame_count - self._window_start
        if window_size == 0 or not self._label:
            return
        width, height = self._last_frame_size
        LOGGER.debug(f"Running SAM3 video propagation on {window_size} frames, prompt={self._label!r}")

        predictor = self._video_predictor
        response = predictor.handle_request(
            dict(
                type="start_session",
                resource_path=self._frame_dir,
                offload_video_to_cpu=self._device != "cuda",
                offload_state_to_cpu=self._device != "cuda",
            )
        )
        session_id = response["session_id"]
        try:
            prompt = dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=self._label,
                output_prob_thresh=self._confidence_threshold,
            )
            if self._initial_point is not None and width > 0 and height > 0:
                # SAM3 add_prompt points default to relative coordinates.
                px, py = self._initial_point
                prompt["points"] = [[px / width, py / height]]
                prompt["point_labels"] = [1]
                prompt["rel_coordinates"] = True
            predictor.handle_request(prompt)

            self._detections = {}
            for streamed in predictor.handle_stream_request(
                dict(type="propagate_in_video", session_id=session_id)
            ):
                local_idx = streamed["frame_index"]
                raw = detections_from_video_outputs(
                    streamed["outputs"],
                    width,
                    height,
                    self._label,
                    min_score=self._confidence_threshold,
                )
                global_idx = self._window_start + int(local_idx)
                dets = [_to_viam(d) for d in raw]
                self._detections[global_idx] = dets
                if dets:
                    self._last_detections = dets
        finally:
            try:
                predictor.handle_request(dict(type="close_session", session_id=session_id))
            except Exception as err:
                LOGGER.debug(f"close_session: {err}")

        self._frames_since_propagation = 0
        LOGGER.debug(
            f"Propagation complete: {sum(1 for v in self._detections.values() if v)}/"
            f"{window_size} frames with detections"
        )

    def _detect_frame(self, image_np: np.ndarray) -> List[Detection]:
        frame_idx = self._save_frame(image_np)
        if self._use_video and self._video_predictor is not None:
            if self._frames_since_propagation >= self._propagation_interval:
                try:
                    self._run_video_propagation()
                except Exception as err:
                    LOGGER.warning(
                        f"Video propagation failed ({err}); using per-frame image detection"
                    )
                    return self._run_image_propagation(image_np, frame_idx)
            dets = self._detections.get(frame_idx)
            if dets is not None:
                return dets
            return list(self._last_detections)
        return self._run_image_propagation(image_np, frame_idx)

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        image_np = viam_image_to_numpy(image)
        with self._lock:
            return self._detect_frame(image_np)

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        image_np = await self._get_camera_image()
        with self._lock:
            return self._detect_frame(image_np)

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> CaptureAllResult:
        if camera_name not in (self._camera_name, ""):
            raise ValueError(
                f"Camera name '{camera_name}' does not match "
                f"configured camera '{self._camera_name}'."
            )

        result = CaptureAllResult()
        images, _ = await self._camera.get_images()
        if images is None or len(images) == 0:
            raise ValueError("No images returned by get_images")

        if return_image:
            result.image = images[0]
        if return_detections:
            result.detections = await self.get_detections(
                images[0], extra=extra, timeout=timeout
            )
        return result

    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        raise NotImplementedError("classifications not supported")

    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        raise NotImplementedError("classifications not supported")

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[PointCloudObject]:
        raise NotImplementedError("object point clouds not supported")

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> Vision.Properties:
        return GetPropertiesResponse(
            classifications_supported=False,
            detections_supported=True,
            object_point_clouds_supported=False,
        )

    async def close(self):
        if self._frame_dir and os.path.isdir(self._frame_dir):
            shutil.rmtree(self._frame_dir, ignore_errors=True)
            LOGGER.debug(f"Cleaned up frame directory: {self._frame_dir}")
        self._frame_dir = None

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        cmd = command.get("command", "")

        if cmd == "set_label":
            label = str(command.get("label", "")).strip()
            if not label:
                return {"error": "label is required, e.g. 'stemless wine glass'"}
            with self._lock:
                self._label = label
                self._detections = {}
                self._last_detections = []
            LOGGER.info(f"Text prompt set to {label!r}")
            return {"status": f"label set to {label!r}"}

        if cmd == "set_point":
            x = int(command["x"])
            y = int(command["y"])
            with self._lock:
                self._initial_point = (x, y)
                self._detections = {}
                self._last_detections = []
            return {"status": f"optional point prompt set to ({x}, {y})"}

        if cmd == "clear_point":
            with self._lock:
                self._initial_point = None
                self._detections = {}
                self._last_detections = []
            return {"status": "optional point prompt cleared"}

        if cmd == "reprocess":
            with self._lock:
                if self._use_video and self._video_predictor is not None:
                    self._run_video_propagation()
                    n = len(self._detections)
                else:
                    n = 0
            return {
                "status": "ok",
                "num_frames": float(self._frame_count - self._window_start),
                "detections": float(n),
            }

        if cmd == "reset":
            with self._lock:
                if self._frame_dir:
                    shutil.rmtree(self._frame_dir, ignore_errors=True)
                    self._frame_dir = tempfile.mkdtemp(prefix="sam3_frames_")
                self._frame_count = 0
                self._window_start = 0
                self._frames_since_propagation = 0
                self._detections = {}
                self._last_detections = []
            return {"status": "reset complete"}

        if cmd == "status":
            return {
                "total_frames_received": float(self._frame_count),
                "window_size": float(self._frame_count - self._window_start),
                "max_frames": float(self._max_frames),
                "detections_cached": float(sum(len(v) for v in self._detections.values())),
                "device": self._device,
                "model_name": SAM3_MODEL_ID,
                "label": self._label,
                "confidence_threshold": float(self._confidence_threshold),
                "use_video": self._use_video,
                "propagation_interval": float(self._propagation_interval),
                **torch_build_info(),
            }

        return {"error": f"unknown command: {cmd}"}
