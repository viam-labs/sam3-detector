# Model viam:sam3-detector:sam3

A Viam vision service that uses Meta's [SAM3](https://github.com/facebookresearch/sam3) to detect every instance of an open-vocabulary concept from a **text label alone**. No bounding box or click is required.

Example: set `label` to `"stemless wine glass"` and `get_detections` returns a box (and confidence) for each matching instance in the frame.

On CUDA the module also runs SAM3's video predictor over a sliding window of recent frames so identities stay stable across time. On CPU/MPS it falls back to per-frame image detection with the same text prompt.

## Configuration

```json
{
  "camera_name": "my-camera",
  "label": "stemless wine glass",
  "confidence_threshold": 0.5,
  "propagation_interval": 1,
  "max_frames": 300
}
```

### Attributes

| Name | Type | Inclusion | Description |
|---|---|---|---|
| `camera_name` | string | **Required** | Camera to use for `get_detections_from_camera`. Added as a required dependency. |
| `label` | string | **Required** | Open-vocabulary text prompt. Short noun phrases work best (`"person"`, `"stemless wine glass"`). This is the SAM3 prompt — not a post-filter on someone else's detector. |
| `confidence_threshold` | float | Optional | Minimum SAM3 score to keep a detection. Default: `0.5`. |
| `initial_point_x` | int | Optional | Extra positive point prompt (pixels). SAM3 does **not** need this; use it only to disambiguate. Must be paired with `initial_point_y`. |
| `initial_point_y` | int | Optional | Extra positive point prompt (pixels). Must be paired with `initial_point_x`. |
| `propagation_interval` | int | Optional | Re-run SAM3 video propagation every N frames (CUDA path). Default: `1`. |
| `max_frames` | int | Optional | Maximum frames to keep in the sliding window. Default: `300`. |

### Example Configuration

```json
{
  "camera_name": "my-camera",
  "label": "stemless wine glass"
}
```

## DoCommand

### `set_label` — Change the text prompt

```json
{
  "command": "set_label",
  "label": "stemless wine glass"
}
```

Sets a new open-vocabulary prompt and clears cached detections. No bounding box is required.

### `set_point` — Optional click prompt

```json
{
  "command": "set_point",
  "x": 400,
  "y": 250
}
```

Adds a geometric point on top of the text prompt (CUDA video path). Call `reprocess` afterward to re-run propagation.

### `clear_point` — Drop the optional click

```json
{
  "command": "clear_point"
}
```

### `reprocess` — Re-run video propagation

```json
{
  "command": "reprocess"
}
```

Re-runs SAM3 video propagation on all frames in the current window (CUDA). Returns `num_frames` and `detections` count.

### `reset` — Clear all state

```json
{
  "command": "reset"
}
```

Clears all buffered frames and cached detections.

### `status` — Get current state

```json
{
  "command": "status"
}
```

Returns: `total_frames_received`, `window_size`, `max_frames`, `detections_cached`, `device`, `model_name`, `label`, `confidence_threshold`, `use_video`, `propagation_interval`, plus `torch_version` and `torch_gpu_support`.
