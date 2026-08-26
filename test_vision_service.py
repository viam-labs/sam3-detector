"""
Test script: connects to a Viam robot running the sam3-detector module,
calls capture_all_from_camera to get detections, and saves annotated frames.

Usage:
    python test_vision_service.py
"""

import asyncio
import io
import os
import subprocess
import time

from PIL import Image, ImageDraw
from viam.robot.client import RobotClient
from viam.rpc.dial import DialOptions
from viam.services.vision import VisionClient

env_path = os.path.join(os.path.dirname(__file__), "..", "viam.env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

ROBOT_ADDRESS = os.environ.get("VIAM_ROBOT_ADDRESS", "vino2-main.kssbd6djf3.viam.cloud")
VISION_SERVICE_NAME = os.environ.get("VIAM_VISION_SERVICE", "vision-sam3")
NUM_FRAMES = 50
OUTPUT_DIR = "test_vision_output"


async def main():
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

    vision = VisionClient.from_robot(robot, VISION_SERVICE_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    status = await vision.do_command({"command": "status"})
    print(f"Module status: {status}")

    print(f"Capturing {NUM_FRAMES} frames from vision service '{VISION_SERVICE_NAME}'...")
    detected_count = 0
    for i in range(NUM_FRAMES):
        t0 = time.time()
        try:
            result = await vision.capture_all_from_camera(
                "",
                return_image=True,
                return_detections=True,
                timeout=60,
            )
        except Exception as e:
            print(f"  Frame {i}: ERROR - {e}")
            continue
        elapsed_ms = (time.time() - t0) * 1000

        pil = None
        if result.image is not None:
            pil = Image.open(io.BytesIO(result.image.data)).convert("RGB")

        dets = result.detections or []
        if dets:
            detected_count += 1

        if pil is not None:
            draw = ImageDraw.Draw(pil)
            draw.text((10, 10), f"#{i}  {elapsed_ms:.0f}ms  dets:{len(dets)}", fill="white")
            for det in dets:
                x0, y0, x1, y1 = det.x_min, det.y_min, det.x_max, det.y_max
                draw.rectangle([x0, y0, x1, y1], outline="lime", width=3)
                label = f"{det.class_name} {det.confidence:.2f}"
                draw.text((x0, y0 - 15), label, fill="lime")
            pil.save(os.path.join(OUTPUT_DIR, f"{i}.jpeg"), quality=90)

        if (i + 1) % 10 == 0:
            print(
                f"  Frame {i + 1}/{NUM_FRAMES}  ({elapsed_ms:.0f}ms)  "
                f"dets: {len(dets)}  total_detected: {detected_count}/{i + 1}"
            )

    await robot.close()

    video_path = "test_vision_tracked.mp4"
    print(f"Building video: {video_path}")
    subprocess.run(
        [
            "ffmpeg", "-y", "-framerate", "5",
            "-i", f"{OUTPUT_DIR}/%d.jpeg",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            video_path,
        ],
        capture_output=True,
    )

    print(f"\nDone! {detected_count}/{NUM_FRAMES} frames with detections")
    print(f"Annotated frames saved to {OUTPUT_DIR}/")
    print(f"Video saved to {video_path}")


if __name__ == "__main__":
    asyncio.run(main())
