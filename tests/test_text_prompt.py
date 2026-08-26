"""DoCommand set_label / status without loading SAM3 weights."""

import asyncio
import threading
from unittest.mock import MagicMock

import numpy as np
from models.boxes import RawDetection
from models.sam3 import Sam3, _to_viam


def test_to_viam_detection_preserves_label():
    raw = RawDetection(
        x_min=1, y_min=2, x_max=3, y_max=4,
        confidence=0.5, class_name="stemless wine glass", object_id=9,
    )
    det = _to_viam(raw)
    assert det.class_name == "stemless wine glass"
    assert det.x_min == 1
    assert det.y_max == 4
    assert det.confidence == 0.5


def test_meta_json_registers_both_models():
    import json
    import os

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    with open(os.path.join(root, "meta.json")) as f:
        meta = json.load(f)
    models = [m["model"] for m in meta["models"]]
    assert "viam:sam3-detector:sam3" in models
    assert "viam:sam3-detector:sam3-segments" in models
    assert meta["module_id"] == "viam:sam3-detector"
    assert meta["entrypoint"] == "run.sh"


def test_image_propagation_uses_text_prompt_not_a_box():
    """Sam3._run_image_on_array must pass the label into set_text_prompt."""
    from models.sam3 import Sam3

    boxes = np.array([[10.0, 20.0, 30.0, 40.0]])
    scores = np.array([0.91])

    processor = MagicMock()
    processor.set_image.return_value = {"backbone_out": {}}
    processor.set_text_prompt.return_value = {
        "boxes": boxes,
        "scores": scores,
        "masks": None,
    }

    inst = Sam3.__new__(Sam3)
    inst._processor = processor
    inst._label = "stemless wine glass"
    inst._confidence_threshold = 0.5

    image = np.zeros((80, 100, 3), dtype=np.uint8)
    dets = inst._run_image_on_array(image)

    processor.set_text_prompt.assert_called_once()
    args, kwargs = processor.set_text_prompt.call_args
    assert args[0] == "stemless wine glass"
    assert len(dets) == 1
    assert dets[0].class_name == "stemless wine glass"
    assert dets[0].x_min == 10
    assert dets[0].y_max == 40


def test_set_label_do_command_changes_prompt_without_a_box():
    inst = Sam3.__new__(Sam3)
    inst._lock = threading.Lock()
    inst._label = "cup"
    inst._detections = {0: ["stale"]}
    inst._last_detections = ["stale"]

    result = asyncio.run(
        inst.do_command({"command": "set_label", "label": "stemless wine glass"})
    )
    assert inst._label == "stemless wine glass"
    assert inst._detections == {}
    assert inst._last_detections == []
    assert "stemless wine glass" in result["status"]


def test_set_label_rejects_empty():
    inst = Sam3.__new__(Sam3)
    inst._lock = threading.Lock()
    inst._label = "cup"
    result = asyncio.run(inst.do_command({"command": "set_label", "label": "  "}))
    assert "error" in result
    assert inst._label == "cup"
