"""camera_names lets one SAM3 model serve several RGBD cameras."""

import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from viam.components.camera import Camera

from models.boxes import RawDetection
from models.sam3_segments import Sam3Segments


class _Str:
    def __init__(self, s):
        self.string_value = s
        self.number_value = 0
        self.bool_value = False


class _List:
    def __init__(self, names):
        self.list_value = SimpleNamespace(values=[_Str(n) for n in names])


def _config(fields):
    return SimpleNamespace(
        name="sam3-vision",
        api="rdk:service:vision",
        attributes=SimpleNamespace(fields=fields),
        log_configuration=SimpleNamespace(level=""),
    )


def _fields(extra=None):
    fields = {"camera_name": _Str("cam"), "label": _Str("chip bag")}
    if extra is not None:
        fields["camera_names"] = _List(extra)
    return fields


class _FakeCamera:
    def __init__(self, name):
        self.name = name
        self.calls = 0

    async def get_images(self):
        self.calls += 1
        return [SimpleNamespace(name="color", data=b"")], None

    async def get_properties(self):
        return SimpleNamespace(
            intrinsic_parameters=SimpleNamespace(
                focal_x_px=1.0, focal_y_px=1.0, center_x_px=0.0, center_y_px=0.0
            )
        )


def _service(extra=None):
    svc = Sam3Segments.__new__(Sam3Segments)
    svc.name = "sam3-vision"
    svc._lock = threading.Lock()
    svc._processor = None
    svc._robot_client = None
    cams = {"cam": _FakeCamera("cam")}
    for name in extra or []:
        cams[name] = _FakeCamera(name)
    deps = {Camera.get_resource_name(n): c for n, c in cams.items()}
    svc.reconfigure(_config(_fields(extra)), deps)
    return svc, cams


def test_validate_adds_extra_cameras_as_dependencies():
    deps, optional = Sam3Segments.validate_config(
        _config(_fields(["cam", "other-rgbd-cam"]))
    )
    assert deps == ["cam", "other-rgbd-cam"]
    assert optional == []


def test_validate_without_camera_names_is_unchanged():
    deps, _ = Sam3Segments.validate_config(_config(_fields()))
    assert deps == ["cam"]


def test_validate_rejects_empty_camera_name_entry():
    with pytest.raises(ValueError, match="camera_names"):
        Sam3Segments.validate_config(_config(_fields(["other-rgbd-cam", ""])))


def test_resolve_camera_picks_requested_and_defaults_on_empty():
    svc, cams = _service(["other-rgbd-cam"])
    assert svc._resolve_camera("other-rgbd-cam") == ("other-rgbd-cam", cams["other-rgbd-cam"])
    assert svc._resolve_camera("") == ("cam", cams["cam"])
    assert svc._resolve_camera("cam") == ("cam", cams["cam"])


def test_resolve_camera_rejects_unlisted_camera():
    svc, _ = _service(["other-rgbd-cam"])
    with pytest.raises(ValueError, match="camera_names"):
        svc._resolve_camera("tag-cam")


def test_point_clouds_use_requested_camera_for_images_and_frame(monkeypatch):
    svc, cams = _service(["other-rgbd-cam"])
    seen = {}

    async def color_and_depth(camera):
        seen["camera"] = camera
        color = np.zeros((4, 4, 3), dtype=np.uint8)
        depth = np.full((4, 4), 500.0)
        return None, color, depth

    async def prompt(image):
        mask = np.ones((4, 4), dtype=bool)
        return [RawDetection(0, 0, 3, 3, 0.9, "chip bag")], [mask]

    async def to_world(points, camera_name):
        seen["frame"] = camera_name
        return points, camera_name

    monkeypatch.setattr(svc, "_get_color_and_depth", color_and_depth)
    monkeypatch.setattr(svc, "_prompt_image", prompt)
    monkeypatch.setattr(svc, "_transform_points_to_world", to_world)
    svc._min_points = 1

    objs = asyncio.run(svc.get_object_point_clouds("other-rgbd-cam"))
    assert len(objs) == 1
    assert seen["camera"] is cams["other-rgbd-cam"]
    assert seen["frame"] == "other-rgbd-cam"
    assert objs[0].geometries.reference_frame == "other-rgbd-cam"


def test_detections_from_camera_uses_requested_camera(monkeypatch):
    svc, cams = _service(["other-rgbd-cam"])

    async def prompt(image):
        return [], []

    monkeypatch.setattr(svc, "_prompt_image", prompt)
    asyncio.run(svc.get_detections_from_camera("other-rgbd-cam"))
    assert cams["other-rgbd-cam"].calls == 1
    assert cams["cam"].calls == 0


def test_status_reports_configured_cameras(monkeypatch):
    monkeypatch.setattr("models.sam3_segments.torch_build_info", lambda: {})
    svc, _ = _service(["other-rgbd-cam"])
    svc._device = "cpu"
    status = asyncio.run(svc.do_command({"command": "status"}))
    assert status["camera_names"] == ["cam", "other-rgbd-cam"]
