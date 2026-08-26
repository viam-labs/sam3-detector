# Model viam:sam3-detector:sam3-segments

A Viam vision service that uses SAM3 text prompts to find object masks, then projects those pixels through an RGBD depth map to produce 3D point clouds.

Unlike `sam2-segments`, this model does **not** need an upstream 2D detector or a bounding box. The `label` attribute is the SAM3 prompt (`"stemless wine glass"`). An optional `detector_name` can still be set; if present, SAM3 instances that do not overlap an upstream box are dropped.

## How it works

1. Gets color and depth images from an RGBD camera
2. Runs SAM3 with the configured text prompt (no box required)
3. Optionally intersects those instances with an upstream detector
4. Projects only the masked pixels to 3D using camera intrinsics and the depth map
5. Skips zero-depth pixels and optionally filters background by depth threshold
6. Returns each segmented object as a `PointCloudObject`

## Configuration

```json
{
  "camera_name": "my-rgbd-camera",
  "label": "stemless wine glass",
  "confidence_threshold": 0.5,
  "depth_threshold_mm": 200,
  "min_points": 50,
  "highlighting_on": true,
  "highlight_color": { "r": 0, "g": 255, "b": 0 }
}
```

### Attributes

| Name | Type | Inclusion | Description |
|---|---|---|---|
| `camera_name` | string | **Required** | Name of the RGBD camera that provides both color and depth images. Added as a required dependency. |
| `label` | string | **Required** | Open-vocabulary text prompt used as the SAM3 query. Default is not applied — you must set this. |
| `detector_name` | string | Optional | If set, SAM3 instances that do not overlap a detection from this vision service are discarded. |
| `confidence_threshold` | float | Optional | Minimum SAM3 (and optional upstream) score. Default: `0.5`. |
| `depth_threshold_mm` | int | Optional | Maximum deviation from median depth (in mm) to keep a point. Default: `0` (disabled). |
| `min_points` | int | Optional | Minimum number of 3D points for a valid segment. Default: `50`. |
| `highlighting_on` | bool | Optional | When true, images returned by `capture_all_from_camera` have SAM3 masks overlaid at 50% opacity. Default: `false`. |
| `highlight_color` | object | Optional | RGB color used for the mask overlay. Default: `{ "r": 0, "g": 255, "b": 0 }`. |
| `debug` | bool | Optional | When true, also return a `debug_<label>_before_cleaning` point cloud. Default: `false`. |

### Example Configuration

```json
{
  "camera_name": "realsense",
  "label": "stemless wine glass",
  "confidence_threshold": 0.3,
  "depth_threshold_mm": 150,
  "min_points": 100
}
```

## Supported API methods

### `get_object_point_clouds(camera_name)`

Returns a list of `PointCloudObject` — one for each text-prompted instance. Each contains:
- `point_cloud`: PCD binary data (XYZRGB format, coordinates in meters)
- `geometries`: 3D bounding box geometry with label (dimensions in mm)

Point clouds are automatically transformed to the **world frame** using the machine's frame system. The module connects to the parent machine using the `VIAM_MACHINE_FQDN`, `VIAM_API_KEY_ID`, and `VIAM_API_KEY` environment variables, which are set automatically by `viam-server`. If frame transform is unavailable, points are returned in the camera's reference frame.

### `get_detections(image)` / `get_detections_from_camera(camera_name)`

Returns SAM3 boxes for the configured text prompt. No upstream bounding box is required.

### `capture_all_from_camera(camera_name)`

Supports `return_image`, `return_detections`, and `return_object_point_clouds`.

### `get_properties()`

Returns `detections_supported=True, object_point_clouds_supported=True`.

## DoCommand

### `set_label`

```json
{
  "command": "set_label",
  "label": "stemless wine glass"
}
```

### `status`

```json
{
  "command": "status"
}
```

Returns current configuration: camera name, optional detector name, label, confidence threshold, model name, device, depth threshold, min points, plus torch GPU build info.

## Camera requirements

The camera must return both **color** and **depth** images from `get_images()`. The images are matched by source name:
- Color: source name containing "color" or "rgb"
- Depth: source name containing "depth"

The camera must also provide **intrinsic parameters** via `get_properties()` (focal length, principal point).

Intel RealSense cameras work well for this purpose.

## Frame transforms

Point clouds from `get_object_point_clouds` are automatically transformed from the camera frame to the **world frame** using the machine's frame system configuration.

The module connects to the parent machine using environment variables set by `viam-server`:
- `VIAM_MACHINE_FQDN` — machine address
- `VIAM_API_KEY_ID` — API key ID
- `VIAM_API_KEY` — API key

These are set automatically — no additional configuration is needed. If the frame system is not configured or the connection fails, the module logs a warning and returns point clouds in the camera's reference frame instead.
