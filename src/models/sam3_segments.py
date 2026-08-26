"""
SAM3 Segments: text-prompted SAM3 masks + depth projection → PointCloudObject.

Unlike sam2-segments (which required an upstream 2D detector to supply a
bounding box), SAM3 finds every instance of a noun phrase such as
"stemless wine glass" from the label alone.
"""

from __future__ import annotations

import io
import os
import struct
import threading
from typing import ClassVar, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image as PILImage
from typing_extensions import Self
from viam.components.camera import Camera
from viam.logging import getLogger
from viam.media.video import CameraMimeType, ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import (
    GeometriesInFrame,
    Geometry,
    PointCloudObject,
    Pose,
    PoseInFrame,
    RectangularPrism,
    ResourceName,
    Vector3,
)
from viam.proto.service.vision import Classification, Detection, GetPropertiesResponse
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.robot.client import RobotClient
from viam.rpc.dial import DialOptions
from viam.services.vision import CaptureAllResult, Vision
from viam.utils import ValueTypes

from models.boxes import RawDetection, detections_from_processor_state
from models.common import (
    SAM3_MODEL_ID,
    numpy_to_pil,
    select_device,
    torch_build_info,
    viam_image_to_numpy,
)
from models.loader import load_image_processor
from spatialmath import transform_points_with_pose

LOGGER = getLogger(__name__)

MAX_DEPTH_MM = 20000


async def _connect_to_machine() -> Optional[RobotClient]:
    """Connect to the parent machine using env vars set by viam-server.
    Returns None if env vars are not set (e.g. running standalone)."""
    host = os.environ.get("VIAM_MACHINE_FQDN", "")
    api_key_id = os.environ.get("VIAM_API_KEY_ID", "")
    api_key = os.environ.get("VIAM_API_KEY", "")
    missing = []
    if not host:
        missing.append("VIAM_MACHINE_FQDN")
    if not api_key_id:
        missing.append("VIAM_API_KEY_ID")
    if not api_key:
        missing.append("VIAM_API_KEY")
    if missing:
        LOGGER.warn(
            f"Frame system unavailable: missing environment variable(s): {', '.join(missing)}. "
            f"Point clouds will be returned in camera frame instead of world frame."
        )
        return None
    try:
        client = await RobotClient.at_address(
            host,
            RobotClient.Options(
                dial_options=DialOptions.with_api_key(api_key=api_key, api_key_id=api_key_id),
            ),
        )
        LOGGER.debug(f"Connected to machine at {host} for frame transforms")
        return client
    except Exception as e:
        LOGGER.warn(f"Could not connect to machine for frame transforms: {e}")
        return None


def _depth_image_to_numpy(image: ViamImage) -> np.ndarray:
    """Convert a Viam depth image to a numpy array (H, W) of depth in mm."""
    mime = getattr(image, "mime_type", "")
    data = image.data
    LOGGER.debug(
        f"Depth image: mime={mime}, data_len={len(data)}, "
        f"first_bytes={data[:24].hex() if len(data) >= 24 else data.hex()}"
    )

    if "viam" in str(mime) and "dep" in str(mime):
        depth_list = image.bytes_to_depth_array()
        return np.array(depth_list, dtype=np.float64)

    pil = PILImage.open(io.BytesIO(data))
    LOGGER.debug(f"Depth PIL: mode={pil.mode}, size={pil.size}")
    return np.array(pil, dtype=np.float64)


def _encode_pcd_binary(points_mm: np.ndarray, colors: np.ndarray) -> bytes:
    """Encode points (N,3) in mm and colors (N,3) uint8 as binary PCD.
    PCD coordinates are in meters (Viam convention)."""
    n = len(points_mm)
    if n == 0:
        return b""

    header = (
        f"VERSION .7\n"
        f"FIELDS x y z rgb\n"
        f"SIZE 4 4 4 4\n"
        f"TYPE F F F F\n"
        f"COUNT 1 1 1 1\n"
        f"WIDTH {n}\n"
        f"HEIGHT 1\n"
        f"VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        f"DATA binary\n"
    ).encode("ascii")

    points_m = points_mm / 1000.0
    buf = bytearray(n * 16)
    for i in range(n):
        x, y, z = points_m[i]
        r, g, b = int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])
        rgb_int = (r << 16) | (g << 8) | b
        rgb_float = struct.unpack("f", struct.pack("I", rgb_int))[0]
        struct.pack_into("ffff", buf, i * 16, float(x), float(y), float(z), rgb_float)

    return header + bytes(buf)


class Sam3Segments(Vision, EasyResource):
    """Vision service: text prompt → SAM3 masks → depth → 3D point clouds."""

    MODEL: ClassVar[Model] = Model(ModelFamily("viam", "sam3-detector"), "sam3-segments")

    _processor: object
    _device: str = "cpu"
    _camera: ResourceBase
    _detector: Optional[ResourceBase] = None
    _detector_name: str = ""
    _camera_name: str = ""
    _label: str = ""
    _confidence_threshold: float = 0.5
    _depth_threshold_mm: int = 0
    _min_points: int = 50
    _highlighting_on: bool = False
    _highlight_color: Tuple[int, int, int] = (0, 255, 0)
    _debug: bool = False
    _robot_client: Optional[RobotClient] = None
    _lock: threading.Lock

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        instance = super().new(config, dependencies)
        instance._lock = threading.Lock()
        attrs = config.attributes.fields

        instance._camera_name = attrs["camera_name"].string_value
        instance._camera = dependencies[Camera.get_resource_name(instance._camera_name)]
        instance._label = attrs["label"].string_value
        LOGGER.info(f"Text prompt (label): {instance._label!r}")

        instance._detector_name = ""
        instance._detector = None
        if "detector_name" in attrs and attrs["detector_name"].string_value:
            instance._detector_name = attrs["detector_name"].string_value
            instance._detector = dependencies[Vision.get_resource_name(instance._detector_name)]
            LOGGER.debug(
                f"Optional upstream detector {instance._detector_name} will filter SAM3 results"
            )

        if "confidence_threshold" in attrs:
            instance._confidence_threshold = attrs["confidence_threshold"].number_value
        if "depth_threshold_mm" in attrs:
            instance._depth_threshold_mm = int(attrs["depth_threshold_mm"].number_value)
        if "min_points" in attrs:
            instance._min_points = int(attrs["min_points"].number_value)
        if "highlighting_on" in attrs:
            instance._highlighting_on = attrs["highlighting_on"].bool_value
        if "highlight_color" in attrs:
            color_fields = attrs["highlight_color"].struct_value.fields
            instance._highlight_color = (
                max(0, min(255, int(color_fields["r"].number_value))),
                max(0, min(255, int(color_fields["g"].number_value))),
                max(0, min(255, int(color_fields["b"].number_value))),
            )
        if "debug" in attrs:
            instance._debug = attrs["debug"].bool_value

        instance._device = select_device()
        LOGGER.debug(f"Loading SAM3 ImageProcessor ({SAM3_MODEL_ID}) on {instance._device}")
        instance._processor = load_image_processor(
            instance._device, confidence_threshold=instance._confidence_threshold
        )
        LOGGER.info("SAM3 ImageProcessor loaded")
        instance._robot_client = None
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
                "prompt (e.g. 'stemless wine glass'). No bounding box or "
                "upstream detector is required."
            )
        deps = [attrs["camera_name"].string_value]
        if "detector_name" in attrs and attrs["detector_name"].string_value:
            deps.append(attrs["detector_name"].string_value)
        return deps, []

    def _sam3_text_prompt(self, color_np: np.ndarray) -> Tuple[List[RawDetection], List[np.ndarray]]:
        """Run SAM3 text prompt. Returns (detections, binary masks aligned 1:1)."""
        pil = numpy_to_pil(color_np)
        with self._lock:
            state = self._processor.set_image(pil)
            state = self._processor.set_text_prompt(self._label, state)

        dets = detections_from_processor_state(
            state, self._label, min_score=self._confidence_threshold
        )
        masks_t = state.get("masks")
        masks: List[np.ndarray] = []
        if masks_t is not None:
            import torch

            if isinstance(masks_t, torch.Tensor):
                arr = masks_t.detach().cpu().numpy()
            else:
                arr = np.asarray(masks_t)
            if arr.ndim == 2:
                arr = arr[None, ...]
            for i in range(len(dets)):
                if i < len(arr):
                    masks.append(np.squeeze(arr[i]).astype(bool))
        # If mask count drifted, derive from boxes via empty skip.
        while len(masks) < len(dets):
            masks.append(np.zeros(color_np.shape[:2], dtype=bool))
        return dets, masks

    async def _maybe_filter_with_upstream(
        self, image: ViamImage, dets: List[RawDetection], masks: List[np.ndarray]
    ) -> Tuple[List[RawDetection], List[np.ndarray]]:
        """If an optional upstream detector is configured, keep SAM3 instances
        that overlap an upstream box of the same (or any) label."""
        if self._detector is None:
            return dets, masks
        upstream = await self._detector.get_detections(image)
        kept_dets: List[RawDetection] = []
        kept_masks: List[np.ndarray] = []
        for det, mask in zip(dets, masks):
            for u in upstream:
                if u.confidence < self._confidence_threshold:
                    continue
                if not _boxes_overlap(
                    (det.x_min, det.y_min, det.x_max, det.y_max),
                    (u.x_min, u.y_min, u.x_max, u.y_max),
                ):
                    continue
                kept_dets.append(det)
                kept_masks.append(mask)
                break
        return kept_dets, kept_masks

    async def _get_color_and_depth(self) -> Tuple[ViamImage, np.ndarray, np.ndarray]:
        images, _ = await self._camera.get_images()
        color_img = None
        depth_img = None
        for img in images:
            name = getattr(img, "name", "") or getattr(img, "source_name", "") or ""
            if "color" in name.lower() or "rgb" in name.lower():
                color_img = img
            elif "depth" in name.lower():
                depth_img = img
        if color_img is None and len(images) >= 1:
            color_img = images[0]
        if depth_img is None and len(images) >= 2:
            depth_img = images[1]
        if color_img is None or depth_img is None:
            raise ValueError("Camera must return both color and depth images")
        return color_img, viam_image_to_numpy(color_img), _depth_image_to_numpy(depth_img)

    def _overlay_masks(self, image: ViamImage, masks: List[np.ndarray]) -> ViamImage:
        color_np = viam_image_to_numpy(image)
        overlay = color_np.copy()
        color = np.array(self._highlight_color, dtype=np.uint8)
        for mask in masks:
            bool_mask = np.asarray(mask).astype(bool)
            if bool_mask.shape[:2] != overlay.shape[:2]:
                continue
            overlay[bool_mask] = (
                overlay[bool_mask].astype(np.float32) * 0.5 + color * 0.5
            ).astype(np.uint8)
        pil = PILImage.fromarray(overlay)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=90)
        return ViamImage(buf.getvalue(), CameraMimeType.JPEG)

    def _mask_to_points_raw(
        self,
        mask: np.ndarray,
        color_np: np.ndarray,
        depth_np: np.ndarray,
        fx: float, fy: float, ppx: float, ppy: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        vs, us = np.where(mask)
        depths = depth_np[vs, us].astype(np.float64)
        valid = (depths > 0) & (depths < MAX_DEPTH_MM)
        vs, us, depths = vs[valid], us[valid], depths[valid]
        if len(depths) == 0:
            return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
        xs = (us.astype(np.float64) - ppx) * depths / fx
        ys = (vs.astype(np.float64) - ppy) * depths / fy
        zs = depths
        points = np.stack([xs, ys, zs], axis=1)
        colors = color_np[vs, us]
        return points, colors

    def _clean_points(
        self, points: np.ndarray, colors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._depth_threshold_mm <= 0 or len(points) == 0:
            return points, colors
        depths = points[:, 2]
        median = np.median(depths)
        within = np.abs(depths - median) <= self._depth_threshold_mm
        return points[within], colors[within]

    def _build_point_cloud_object(
        self, points: np.ndarray, colors: np.ndarray, label: str, ref_frame: str = ""
    ) -> PointCloudObject:
        pcd_bytes = _encode_pcd_binary(points, colors)
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        center = (mins + maxs) / 2.0
        dims = maxs - mins
        geometry = Geometry(
            center=Pose(
                x=float(center[0]), y=float(center[1]), z=float(center[2]),
                o_x=0, o_y=0, o_z=1, theta=0,
            ),
            box=RectangularPrism(dims_mm=Vector3(
                x=float(dims[0]), y=float(dims[1]), z=float(dims[2]),
            )),
            label=label,
        )
        return PointCloudObject(
            point_cloud=pcd_bytes,
            geometries=GeometriesInFrame(
                reference_frame=ref_frame,
                geometries=[geometry],
            ),
        )

    async def _ensure_robot_client(self):
        if self._robot_client is None:
            self._robot_client = await _connect_to_machine()
        return self._robot_client

    async def _transform_points_to_world(self, points_mm: np.ndarray) -> Tuple[np.ndarray, str]:
        client = await self._ensure_robot_client()
        if client is None:
            return points_mm, self._camera_name
        try:
            origin = PoseInFrame(
                reference_frame=self._camera_name,
                pose=Pose(x=0, y=0, z=0, o_x=0, o_y=0, o_z=1, theta=0),
            )
            world_pose_in_frame = await client.transform_pose(origin, "world")
            p = world_pose_in_frame.pose
            LOGGER.debug(
                f"Frame transform {self._camera_name} -> world: "
                f"translation=({p.x:.1f},{p.y:.1f},{p.z:.1f})mm "
                f"OV=({p.o_x:.4f},{p.o_y:.4f},{p.o_z:.4f}) theta={p.theta:.4f}"
            )
            transformed = transform_points_with_pose(
                p.o_x, p.o_y, p.o_z, p.theta,
                p.x, p.y, p.z,
                points_mm,
            )
            LOGGER.debug(f"Transformed {len(points_mm)} points to world frame")
            return transformed, "world"
        except Exception as e:
            LOGGER.warn(f"Could not transform to world frame: {e}")
            return points_mm, self._camera_name

    async def _prompt_image(self, image: ViamImage) -> Tuple[List[RawDetection], List[np.ndarray]]:
        color_np = viam_image_to_numpy(image)
        dets, masks = self._sam3_text_prompt(color_np)
        return await self._maybe_filter_with_upstream(image, dets, masks)

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[PointCloudObject]:
        props = await self._camera.get_properties()
        intrinsics = props.intrinsic_parameters
        if intrinsics is None:
            raise ValueError("Camera must provide intrinsic parameters for 3D projection")
        fx = intrinsics.focal_x_px
        fy = intrinsics.focal_y_px
        ppx = intrinsics.center_x_px
        ppy = intrinsics.center_y_px
        LOGGER.debug(f"Intrinsics: fx={fx}, fy={fy}, ppx={ppx}, ppy={ppy}")

        color_viam, color_np, depth_np = await self._get_color_and_depth()
        dets, masks = await self._prompt_image(color_viam)
        LOGGER.debug(f"SAM3 text prompt {self._label!r} returned {len(dets)} instance(s)")

        results = []
        for det, mask in zip(dets, masks):
            if not np.any(mask):
                bbox = (det.x_min, det.y_min, det.x_max, det.y_max)
                LOGGER.debug(f"Empty mask for {bbox}, skipping")
                continue
            points_raw, colors_raw = self._mask_to_points_raw(
                mask, color_np, depth_np, fx, fy, ppx, ppy
            )
            points, colors = self._clean_points(points_raw, colors_raw)
            if len(points) < self._min_points:
                LOGGER.debug(f"Skipping segment with {len(points)} points (min={self._min_points})")
                continue
            label = det.class_name or self._label or "object"
            cam_center = points.mean(axis=0)
            LOGGER.debug(
                f"Segment '{label}': {len(points)} pts, "
                f"camera frame center=({cam_center[0]:.0f},{cam_center[1]:.0f},{cam_center[2]:.0f})mm"
            )
            points, ref_frame = await self._transform_points_to_world(points)
            world_center = points.mean(axis=0)
            LOGGER.debug(
                f"  -> {ref_frame} frame center="
                f"({world_center[0]:.0f},{world_center[1]:.0f},{world_center[2]:.0f})mm"
            )
            results.append(self._build_point_cloud_object(points, colors, label, ref_frame))
            if self._debug:
                points_raw_world, raw_ref_frame = await self._transform_points_to_world(points_raw)
                results.append(
                    self._build_point_cloud_object(
                        points_raw_world, colors_raw, f"debug_{label}_before_cleaning", raw_ref_frame
                    )
                )
        LOGGER.debug(f"Returning {len(results)} point cloud objects")
        return results

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        dets, _ = await self._prompt_image(image)
        return [
            Detection(
                x_min=d.x_min, y_min=d.y_min, x_max=d.x_max, y_max=d.y_max,
                confidence=d.confidence, class_name=d.class_name,
            )
            for d in dets
        ]

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        images, _ = await self._camera.get_images()
        if not images:
            return []
        return await self.get_detections(images[0], extra=extra, timeout=timeout)

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
                f"Camera name '{camera_name}' does not match configured camera '{self._camera_name}'."
            )
        result = CaptureAllResult()
        images, _ = await self._camera.get_images()
        if not images:
            return result
        color_viam = images[0]
        dets, masks = ([], [])
        if return_image or return_detections:
            dets, masks = await self._prompt_image(color_viam)
        if return_image:
            if self._highlighting_on and masks:
                result.image = self._overlay_masks(color_viam, masks)
            else:
                result.image = color_viam
        if return_detections:
            result.detections = [
                Detection(
                    x_min=d.x_min, y_min=d.y_min, x_max=d.x_max, y_max=d.y_max,
                    confidence=d.confidence, class_name=d.class_name,
                )
                for d in dets
            ]
        if return_object_point_clouds:
            result.objects = await self.get_object_point_clouds(
                camera_name, extra=extra, timeout=timeout
            )
        return result

    async def get_classifications_from_camera(
        self, camera_name: str, count: int, *, extra=None, timeout=None
    ) -> List[Classification]:
        raise NotImplementedError("classifications not supported")

    async def get_classifications(
        self, image: ViamImage, count: int, *, extra=None, timeout=None
    ) -> List[Classification]:
        raise NotImplementedError("classifications not supported")

    async def get_properties(self, *, extra=None, timeout=None) -> Vision.Properties:
        return GetPropertiesResponse(
            classifications_supported=False,
            detections_supported=True,
            object_point_clouds_supported=True,
        )

    async def close(self):
        if self._robot_client is not None:
            await self._robot_client.close()
            self._robot_client = None
        LOGGER.info("Sam3Segments shut down")

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
            LOGGER.info(f"Text prompt set to {label!r}")
            return {"status": f"label set to {label!r}"}
        if cmd == "status":
            return {
                "detector_name": self._detector_name,
                "camera_name": self._camera_name,
                "label": self._label,
                "confidence_threshold": self._confidence_threshold,
                "model_name": SAM3_MODEL_ID,
                "device": self._device,
                "depth_threshold_mm": float(self._depth_threshold_mm),
                "min_points": float(self._min_points),
                **torch_build_info(),
            }
        return {"error": f"unknown command: {cmd}"}


def _boxes_overlap(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)
