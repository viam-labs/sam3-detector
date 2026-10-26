"""Unit tests for SAM3 → Viam detection conversion. No SAM3 weights required."""

import numpy as np

from models.boxes import (
    as_numpy,
    detections_from_processor_state,
    detections_from_video_outputs,
    detections_from_xyxy,
    rel_xywh_to_xyxy,
)


def test_stemless_wine_glass_xyxy_to_detection():
    boxes = np.array([[40.2, 80.8, 140.4, 280.6]], dtype=np.float32)
    scores = np.array([0.93], dtype=np.float32)
    dets = detections_from_xyxy(boxes, scores, "stemless wine glass")
    assert len(dets) == 1
    det = dets[0]
    assert det.class_name == "stemless wine glass"
    assert det.x_min == 40
    assert det.y_min == 81
    assert det.x_max == 140
    assert det.y_max == 281
    assert abs(det.confidence - 0.93) < 1e-5


def test_multiple_instances_same_label():
    boxes = np.array(
        [
            [10, 10, 50, 50],
            [200, 100, 260, 180],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.7], dtype=np.float32)
    dets = detections_from_xyxy(boxes, scores, "stemless wine glass")
    assert len(dets) == 2
    assert all(d.class_name == "stemless wine glass" for d in dets)
    assert dets[0].object_id == 0
    assert dets[1].object_id == 1


def test_confidence_filter():
    boxes = np.array([[0, 0, 10, 10], [20, 20, 40, 40]], dtype=np.float32)
    scores = np.array([0.2, 0.8], dtype=np.float32)
    dets = detections_from_xyxy(boxes, scores, "cup", min_score=0.5)
    assert len(dets) == 1
    assert abs(dets[0].confidence - 0.8) < 1e-6


def test_empty_boxes():
    assert detections_from_xyxy(np.zeros((0, 4)), np.zeros((0,)), "x") == []
    assert detections_from_xyxy(None, None, "x") == []


def test_invalid_box_dropped():
    boxes = np.array([[50, 50, 40, 90]], dtype=np.float32)  # x_max < x_min
    dets = detections_from_xyxy(boxes, np.array([0.9]), "x")
    assert dets == []


def test_rel_xywh_in_unit_interval():
    x0, y0, x1, y1 = rel_xywh_to_xyxy([0.1, 0.2, 0.25, 0.4], width=200, height=100)
    assert abs(x0 - 20.0) < 1e-9
    assert abs(y0 - 20.0) < 1e-9
    assert abs(x1 - 70.0) < 1e-9
    assert abs(y1 - 60.0) < 1e-9


def test_abs_xywh_when_values_exceed_one():
    x0, y0, x1, y1 = rel_xywh_to_xyxy([10, 20, 30, 40], width=200, height=100)
    assert (x0, y0, x1, y1) == (10.0, 20.0, 40.0, 60.0)


def test_processor_state_prefers_boxes_over_masks():
    h, w = 32, 48
    mask = np.zeros((1, h, w), dtype=bool)
    mask[0, 2:10, 4:12] = True
    state = {
        "boxes": np.array([[5.0, 6.0, 25.0, 26.0]]),
        "scores": np.array([0.88]),
        "masks": mask,
    }
    dets = detections_from_processor_state(state, "stemless wine glass")
    assert len(dets) == 1
    assert (dets[0].x_min, dets[0].y_min, dets[0].x_max, dets[0].y_max) == (5, 6, 25, 26)


def test_processor_state_falls_back_to_mask_bbox():
    h, w = 32, 48
    mask = np.zeros((1, h, w), dtype=bool)
    mask[0, 2:10, 4:12] = True
    state = {"boxes": None, "scores": np.array([0.77]), "masks": mask}
    dets = detections_from_processor_state(state, "mug")
    assert len(dets) == 1
    assert dets[0].x_min == 4
    assert dets[0].y_min == 2
    assert dets[0].x_max == 11
    assert dets[0].y_max == 9
    assert dets[0].class_name == "mug"


def test_video_outputs_relative_xywh():
    outputs = {
        "out_boxes_xywh": np.array([[0.1, 0.2, 0.2, 0.3]], dtype=np.float32),
        "out_probs": np.array([0.81], dtype=np.float32),
        "out_obj_ids": np.array([7]),
    }
    dets = detections_from_video_outputs(outputs, width=100, height=200, label="person")
    assert len(dets) == 1
    assert dets[0].object_id == 7
    assert dets[0].class_name == "person"
    assert dets[0].x_min == 10
    assert dets[0].y_min == 40
    assert dets[0].x_max == 30
    assert dets[0].y_max == 100


def test_as_numpy_upcasts_bfloat16():
    import pytest

    torch = pytest.importorskip("torch")
    t = torch.tensor([1.25, 2.5], dtype=torch.bfloat16)
    arr = as_numpy(t)
    assert arr.dtype == np.float32
    assert abs(float(arr[0]) - 1.25) < 0.02
