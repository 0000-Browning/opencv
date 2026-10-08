from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import cv2
import numpy as np

if __package__:
    from .calibrate import capture_calibration, inspect_calibration
    from .feature_engine import FeatureEngine
    from .initializer import collect_initial_frames, initialize_two_view
    from .models import Camera, Frame
    from .scale_aligner import select_ruler_points
    from .system import SlamSystem
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from Slam.calibrate import capture_calibration, inspect_calibration
    from Slam.feature_engine import FeatureEngine
    from Slam.initializer import collect_initial_frames, initialize_two_view
    from Slam.models import Camera, Frame
    from Slam.scale_aligner import select_ruler_points
    from Slam.system import SlamSystem


def main() -> None:
    parser = argparse.ArgumentParser(description="Classical monocular feature-based SLAM.")
    parser.add_argument(
        "mode",
        choices=("calibrate", "run", "inspect", "two-view"),
        help="Select camera calibration, live SLAM, inspection, or offline two-view reconstruction.",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--calibration", default="camera_matrix.json")
    parser.add_argument(
        "--board-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(8, 6),
        help="Chessboard inner corner counts for calibration (default: 8 6).",
    )
    parser.add_argument("--scale", type=float, default=0.30, help="Known ruler length in meters.")
    parser.add_argument(
        "--skip-scale",
        action="store_true",
        help="Skip ruler measurement and keep the reconstruction in arbitrary monocular units.",
    )
    parser.add_argument("--bundle-adjustment", action="store_true", help="Enable SciPy local bundle adjustment.")
    parser.add_argument("--image", help="Image path (required by inspect mode).")
    parser.add_argument("--image0", help="First image for two-view mode.")
    parser.add_argument("--image1", help="Second image for two-view mode.")
    parser.add_argument("--ply", default="initial_map.ply", help="Point-cloud output path for two-view mode.")
    parser.add_argument("--output", help="Optional undistorted image path for inspect mode.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.mode == "calibrate":
        _, rms = capture_calibration(
            args.calibration,
            camera_index=args.camera,
            board_size=tuple(args.board_size),
        )
        print(f"Calibration saved to {args.calibration} (RMS error {rms:.3f}px).")
        return
    camera = Camera.load(args.calibration)
    if args.mode == "inspect":
        if not args.image:
            parser.error("--image is required for inspect mode.")
        inspect_calibration(camera, args.image, args.output)
        return

    features = FeatureEngine()
    if args.mode == "two-view":
        if not args.image0 or not args.image1:
            parser.error("--image0 and --image1 are required for two-view mode.")
        image0 = cv2.imread(args.image0)
        image1 = cv2.imread(args.image1)
        if image0 is None or image1 is None:
            raise OSError("Could not read one or both two-view input images.")
        first = features.extract(Frame(0, np.asarray(image0, dtype=np.uint8), camera))
        second = features.extract(Frame(1, np.asarray(image1, dtype=np.uint8), camera))
        slam_map, _ = initialize_two_view(first, second, features)
        if not args.skip_scale:
            select_ruler_points(slam_map, slam_map.keyframes[first.id], args.scale)
        slam_map.save_ply(Path(args.ply))
        print(f"Saved {len(slam_map.points)} scaled map points to {args.ply}.")
        return

    while True:
        first, second = collect_initial_frames(camera, args.camera)
        try:
            slam_map, _ = initialize_two_view(first, second, features)
            break
        except RuntimeError as exc:
            print(f"Initialization failed: {exc} Capture a more textured scene with greater camera translation.")
    if args.skip_scale:
        print("Skipping metric scale alignment; map distances will be in arbitrary units.")
    else:
        scale = select_ruler_points(slam_map, slam_map.keyframes[first.id], args.scale)
        print(f"Metric scale factor: {scale:.6f}")
    SlamSystem(
        camera, slam_map, camera_index=args.camera,
        enable_bundle_adjustment=args.bundle_adjustment,
    ).run()


if __name__ == "__main__":
    main()
