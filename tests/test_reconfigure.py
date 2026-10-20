"""Dependency reconfigure must not KeyError or reload SAM3."""

import asyncio
import logging
import shutil
import threading
from types import SimpleNamespace

from viam.components.camera import Camera
from viam.errors import ResourceNotFoundError
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.manager import ResourceManager
from viam.services.vision import Vision

import sdk_reconfigure
from models.sam3 import Sam3
from models.sam3_segments import Sam3Segments


class _Num:
    def __init__(self, n):
        self.number_value = n
        self.string_value = ""
        self.bool_value = False


class _Str:
    def __init__(self, s):
        self.string_value = s
        self.number_value = 0
        self.bool_value = False


def _config(fields):
    return SimpleNamespace(
        name="sam3-segmenter-left",
        api="rdk:service:vision",
        attributes=SimpleNamespace(fields=fields),
        log_configuration=SimpleNamespace(level=""),
    )


def _sam3_fields(camera="cam-left"):
    return {
        "camera_name": _Str(camera),
        "label": _Str("stemless wine glass"),
        "confidence_threshold": _Num(0.4),
    }


class _Closable:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


def test_remove_missing_resource_does_not_raise():
    """The KeyError from the vino2 log: remove sam2-segmenter-right while it is not registered."""
    sdk_reconfigure.install()
    manager = ResourceManager()
    missing = ResourceName(
        namespace="rdk",
        type="service",
        subtype="vision",
        name="sam2-segmenter-right",
    )
    asyncio.run(manager.remove_resource(missing))
    assert missing not in manager.resources


def test_remove_resource_closes_and_drops_aliases():
    sdk_reconfigure.install()
    manager = ResourceManager()
    resource = _Closable()
    primary = ResourceName(
        namespace="rdk", type="service", subtype="vision", name="sam3-segmenter-left"
    )
    alias = ResourceName(
        namespace="rdk", type="service", subtype="vision", name="sam3-segmenter-left-alias"
    )
    manager.resources[primary] = resource
    manager.resources[alias] = resource
    asyncio.run(manager.remove_resource(primary))
    assert resource.closed
    assert manager.resources == {}


class _Server:
    def __init__(self, resource):
        self.resource = resource
        self.removed = []

    def get_resource(self, of_type, name):
        if self.resource is None:
            raise ResourceNotFoundError("vision", name.name)
        return self.resource

    async def remove_resource(self, name):
        self.removed.append(name)


class _Host:
    def __init__(self, resource):
        self.server = _Server(resource)
        self.added = None

    async def _get_dependencies(self, dependencies):
        return {"deps": list(dependencies)}

    async def add_resource(self, request, deadline=None):
        self.added = request


class _InPlace:
    def __init__(self):
        self.logger = logging.getLogger("test-reconfigure")
        self.seen = None

    def reconfigure(self, config, dependencies):
        self.seen = (config, dependencies)


def test_reconfigure_updates_in_place_and_does_not_remove():
    sdk_reconfigure.install()
    resource = _InPlace()
    host = _Host(resource)
    request = SimpleNamespace(
        config=_config(_sam3_fields()),
        dependencies=["rdk:component:camera/cam-left"],
    )
    asyncio.run(sdk_reconfigure._reconfigure_resource(host, request))
    assert resource.seen is not None
    assert resource.seen[1]["deps"] == ["rdk:component:camera/cam-left"]
    assert host.server.removed == []
    assert host.added is None


def test_reconfigure_readds_when_resource_was_already_removed():
    sdk_reconfigure.install()
    host = _Host(None)
    request = SimpleNamespace(
        config=ComponentConfig(
            name="sam3-segmenter-left",
            api="rdk:service:vision",
            model="viam:sam3-detector:sam3-segments",
        ),
        dependencies=["rdk:component:camera/cam-left"],
    )
    asyncio.run(sdk_reconfigure._reconfigure_resource(host, request))
    assert host.added is not None
    assert host.added.config.name == "sam3-segmenter-left"


def _bare_sam3():
    svc = Sam3.__new__(Sam3)
    svc.name = "sam3-segmenter-left"
    svc._lock = threading.Lock()
    svc._detections = {1: ["stale"]}
    svc._last_detections = ["stale"]
    svc._frame_count = 4
    svc._window_start = 1
    svc._frames_since_propagation = 2
    svc._frame_dir = None
    svc._camera = object()
    svc._camera_name = "cam-left"
    svc._label = "cup"
    svc._confidence_threshold = 0.5
    svc._max_frames = 300
    svc._propagation_interval = 1
    svc._initial_point = None
    svc._processor = SimpleNamespace(confidence_threshold=0.5)
    svc._video_predictor = object()
    return svc


def test_sam3_reconfigure_swaps_camera_and_keeps_the_model():
    svc = _bare_sam3()
    processor = svc._processor
    predictor = svc._video_predictor
    camera = object()
    deps = {Camera.get_resource_name("cam-left"): camera}
    try:
        svc.reconfigure(_config(_sam3_fields()), deps)
        assert svc._camera is camera
        assert svc._label == "stemless wine glass"
        assert svc._confidence_threshold == 0.4
        assert svc._processor is processor
        assert processor.confidence_threshold == 0.4
        assert svc._video_predictor is predictor
        assert svc._detections == {}
        assert svc._frame_count == 0
        assert svc._frame_dir
    finally:
        if svc._frame_dir:
            shutil.rmtree(svc._frame_dir, ignore_errors=True)


def test_segments_reconfigure_swaps_dependencies_and_keeps_processor():
    svc = Sam3Segments.__new__(Sam3Segments)
    svc.name = "sam3-segmenter-left"
    svc._lock = threading.Lock()
    svc._camera = object()
    svc._camera_name = "old"
    svc._label = "cup"
    svc._detector = object()
    svc._detector_name = "old-detector"
    svc._confidence_threshold = 0.5
    svc._depth_threshold_mm = 10
    svc._min_points = 50
    svc._highlighting_on = True
    svc._highlight_color = (1, 2, 3)
    svc._debug = True
    processor = SimpleNamespace(confidence_threshold=0.5)
    svc._processor = processor
    robot = object()
    svc._robot_client = robot

    camera = object()
    deps = {Camera.get_resource_name("cam-right"): camera}
    svc.reconfigure(_config(_sam3_fields("cam-right")), deps)

    assert svc._camera is camera
    assert svc._camera_name == "cam-right"
    assert svc._detector is None
    assert svc._detector_name == ""
    assert svc._highlighting_on is False
    assert svc._processor is processor
    assert svc._robot_client is robot
    assert processor.confidence_threshold == 0.4


def test_segments_reconfigure_keeps_optional_detector():
    svc = Sam3Segments.__new__(Sam3Segments)
    svc.name = "sam2-segmenter-right"
    svc._lock = threading.Lock()
    svc._processor = None
    svc._robot_client = object()
    camera = object()
    detector = object()
    fields = _sam3_fields("cam-right")
    fields["detector_name"] = _Str("yolo")
    deps = {
        Camera.get_resource_name("cam-right"): camera,
        Vision.get_resource_name("yolo"): detector,
    }
    svc.reconfigure(_config(fields), deps)
    assert svc._camera is camera
    assert svc._detector is detector
    assert svc._detector_name == "yolo"
