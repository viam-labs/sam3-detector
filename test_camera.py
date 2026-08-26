"""
Test script: connects to a Viam robot, pulls frames from a camera,
runs SAM3 text-prompt detection (no bounding box), and saves annotated output.

Usage:
    PYTHONPATH=src python test_camera.py [--num-frames 20] [--label "stemless wine glass"]
"""

import asyncio
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw
from viam.components.camera import Camera
from viam.robot.client import RobotClient
from viam.rpc.dial import DialOptions

from models.boxes import detections_from_processor_state
from models.common import numpy_to_pil, select_device
from models.loader import load_image_processor

env_path = os.path.join(os.path.dirname(__file__), "..", "viam.env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

ROBOT_ADDRESS = os.environ.get("VIAM_ROBOT_ADDRESS", "computer-demo-main.496koy7yd1.viam.cloud")
CAMERA_NAME = os.environ.get("VIAM_CAMERA_NAME", "camera-image-dir")
NUM_FRAMES = 20
LABEL = "stemless wine glass"
RAW_DIR = "test_raw_frames"
OUTPUT_DIR = "test_output"


async def main():
    num_frames = NUM_FRAMES
    label = LABEL
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--num-frames" and i < len(sys.argv):
            num_frames = int(sys.argv[i + 1])
        if arg == "--label" and i < len(sys.argv):
            label = sys.argv[i + 1]

    api_key = os.environ.get("VIAM_API_KEY", "")
    api_key_id = os.environ.get("VIAM_API_KEY_ID", "")
    if not api_key or not api_key_id:
        print("Set VIAM_API_KEY and VIAM_API_KEY_ID (or put them in ../viam.env)")
        return

    print(f"Connecting to {ROBOT_ADDRESS}...")
    robot = await RobotClient.at_address(
        ROBOT_ADDRESS,
        RobotClient.Options(
            dial_options=DialOptions.with_api_key(
                api_key=api_key, api_key_id=api_key_id,
            ),
        ),
    )
    print(f"Connected! Resources: {[r.name for r in robot.resource_names]}")

    camera = Camera.from_robot(robot, CAMERA_NAME)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Pulling {num_frames} frames from '{CAMERA_NAME}'...")
    raw_frames = []
    for i in range(num_frames):
        images, _ = await camera.get_images()
        pil = Image.open(io.BytesIO(images[0].data)).convert("RGB")
        arr = np.array(pil)
        raw_frames.append(arr)
        pil.save(os.path.join(RAW_DIR, f"{i}.jpg"), quality=90)
        if i == 0:
            print(f"  Frame size: {arr.shape}")
        if (i + 1) % 10 == 0:
            print(f"  Pulled {i + 1}/{num_frames}")

    await robot.close()

    device = select_device()
    print(f"\nLoading SAM3 image processor on {device}...")
    print(f"Text prompt (no bounding box): {label!r}")
    processor = load_image_processor(device, confidence_threshold=0.5)

    print("Running per-frame text prompts...")
    t0 = time.time()
    results = []
    for i, arr in enumerate(raw_frames):
        state = processor.set_image(numpy_to_pil(arr))
        state = processor.set_text_prompt(label, state)
        dets = detections_from_processor_state(state, label)
        results.append(dets)
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{num_frames}  last frame: {len(dets)} det(s)")
    elapsed = time.time() - t0
    detected = sum(1 for d in results if d)
    print(f"  {elapsed:.1f}s, {detected}/{num_frames} frames with detections")

    print(f"\nSaving annotated frames to {OUTPUT_DIR}/...")
    for i, arr in enumerate(raw_frames):
        pil = Image.fromarray(arr)
        draw = ImageDraw.Draw(pil)
        draw.text((10, 10), f"#{i}  {label}", fill="white")
        for det in results[i]:
            draw.rectangle(
                [det.x_min, det.y_min, det.x_max, det.y_max],
                outline="lime",
                width=3,
            )
            draw.text(
                (det.x_min, max(0, det.y_min - 15)),
                f"{det.class_name} {det.confidence:.2f}",
                fill="lime",
            )
        if not results[i]:
            draw.text((10, 30), "NO DETECTION", fill="red")
        pil.save(os.path.join(OUTPUT_DIR, f"{i}.jpg"), quality=90)

    video_path = "test_tracked.mp4"
    print(f"Building video: {video_path}")
    subprocess.run(
        [
            "ffmpeg", "-y", "-framerate", "5",
            "-i", f"{OUTPUT_DIR}/%d.jpg",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            video_path,
        ],
        capture_output=True,
    )
    print(f"\nDone! Prompt={label!r}  {detected}/{num_frames} frames with detections")


if __name__ == "__main__":
    asyncio.run(main())
