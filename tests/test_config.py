"""Config validation: a text label is required; a bounding box is not."""

from types import SimpleNamespace

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


def _config(fields: dict):
    return SimpleNamespace(attributes=SimpleNamespace(fields=fields))


def test_sam3_requires_camera_and_label_not_a_box():
    cfg = _config(
        {
            "camera_name": _Str("cam"),
            "label": _Str("stemless wine glass"),
        }
    )
    deps, optional = Sam3.validate_config(cfg)
    assert deps == ["cam"]
    assert optional == []


def test_sam3_rejects_missing_label():
    cfg = _config({"camera_name": _Str("cam")})
    try:
        Sam3.validate_config(cfg)
        assert False, "expected ValueError"
    except ValueError as err:
        assert "stemless wine glass" in str(err)
        assert "bounding box" in str(err).lower() or "text" in str(err).lower()


def test_sam3_rejects_partial_optional_point():
    cfg = _config(
        {
            "camera_name": _Str("cam"),
            "label": _Str("cup"),
            "initial_point_x": _Num(10),
        }
    )
    try:
        Sam3.validate_config(cfg)
        assert False, "expected ValueError"
    except ValueError as err:
        assert "initial_point" in str(err)


def test_sam3_segments_does_not_require_detector():
    cfg = _config(
        {
            "camera_name": _Str("realsense"),
            "label": _Str("stemless wine glass"),
        }
    )
    deps, _ = Sam3Segments.validate_config(cfg)
    assert deps == ["realsense"]


def test_sam3_segments_optional_detector_is_a_dependency():
    cfg = _config(
        {
            "camera_name": _Str("realsense"),
            "label": _Str("cup"),
            "detector_name": _Str("yolo"),
        }
    )
    deps, _ = Sam3Segments.validate_config(cfg)
    assert deps == ["realsense", "yolo"]
