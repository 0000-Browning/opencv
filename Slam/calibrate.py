from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import cv2
import numpy as np

from .models import Camera


def validate_board_size(board_size: tuple[int, int]) -> None:
    if len(board_size) != 2 or any(dimension < 2 for dimension in board_size):
        raise ValueError("Board size must specify at least 2 inner corners in each dimension.")


def calibrate_images(
    paths: list[str],
    output: str | Path,
    board_size: tuple[int, int] = (8, 6),
    square_size: float = 1.0,
) -> tuple[Camera, float]:
    validate_board_size(board_size)
    if not math.isfinite(square_size) or square_size <= 0:
        raise ValueError("Chessboard square size must be finite and positive.")
    object_template = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    object_template[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)
    object_template *= square_size
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for image_path in paths:
        image = cv2.imread(image_path)
        if image is None:
            raise OSError(f"Could not read calibration image: {image_path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, board_size)
        if found:
            refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            object_points.append(object_template.copy())
            image_points.append(refined)
            image_size = (gray.shape[1], gray.shape[0])

    if len(image_points) < 10 or image_size is None:
        raise ValueError(f"Calibration needs at least 10 usable views; found {len(image_points)}.")
    rms, matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, np.eye(3, dtype=np.float64),
        np.zeros((5, 1), dtype=np.float64),
    )
    camera = Camera(
        np.asarray(matrix, dtype=np.float64),
        np.asarray(distortion, dtype=np.float64),
    )
    camera.save(output, image_size, rms)
    return camera, float(rms)


def capture_calibration(
    output: str | Path,
    camera_index: int = 0,
    board_size: tuple[int, int] = (8, 6),
    target_views: int = 20,
    square_size: float = 1.0,
) -> tuple[Camera, float]:
    validate_board_size(board_size)
    if target_views < 10:
        raise ValueError("Calibration requires at least 10 target views.")
    if not math.isfinite(square_size) or square_size <= 0:
        raise ValueError("Chessboard square size must be finite and positive.")
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}.")
    object_template = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    object_template[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)
    object_template *= square_size
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    try:
        while len(image_points) < target_views:
            success, image = capture.read()
            if not success:
                raise RuntimeError("Failed to read a frame during camera calibration.")
            if image is None:
                raise RuntimeError("Camera returned an empty calibration frame.")
            image = np.asarray(image, dtype=np.uint8)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, board_size)
            display = image.copy()
            if found and corners is not None:
                cv2.drawChessboardCorners(display, board_size, corners, found)
            cv2.putText(
                display,
                f"{board_size[0]}x{board_size[1]} inner corners | "
                f"Views {len(image_points)}/{target_views} | Space: capture | Q: quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
            )
            cv2.imshow("Camera calibration", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and found and corners is not None:
                object_points.append(object_template.copy())
                image_points.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria))
                image_size = (gray.shape[1], gray.shape[0])
        if len(image_points) < 10 or image_size is None:
            raise ValueError(f"Calibration needs at least 10 views; captured {len(image_points)}.")
        rms, matrix, distortion, _, _ = cv2.calibrateCamera(
            object_points, image_points, image_size, np.eye(3, dtype=np.float64),
            np.zeros((5, 1), dtype=np.float64),
        )
        camera = Camera(
            np.asarray(matrix, dtype=np.float64),
            np.asarray(distortion, dtype=np.float64),
        )
        camera.save(output, image_size, rms)
        return camera, float(rms)
    finally:
        capture.release()
        cv2.destroyWindow("Camera calibration")


def inspect_calibration(camera: Camera, image_path: str | Path, output: str | Path | None = None) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise OSError(f"Could not read image: {image_path}")
    image = np.asarray(image, dtype=np.uint8)
    undistorted = cv2.undistort(image, camera.K, camera.dist)
    if output is not None:
        if not cv2.imwrite(str(output), undistorted):
            raise OSError(f"Could not write undistorted image: {output}")
    cv2.imshow("Original", image)
    cv2.imshow("Undistorted", undistorted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a camera using a chessboard.")
    parser.add_argument("--output", default="camera_matrix.json")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--views", type=int, default=20)
    parser.add_argument(
        "--board-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(8, 6),
        help="Number of inner corner intersections across and down (default: 8 6).",
    )
    parser.add_argument("--square-size", type=float, default=1.0, help="Chessboard square size in chosen units.")
    parser.add_argument("--images", help="Glob for saved calibration images instead of using a webcam.")
    args = parser.parse_args()
    board_size = tuple(args.board_size)
    if args.images:
        camera, rms = calibrate_images(
            glob.glob(args.images), args.output, board_size=board_size,
            square_size=args.square_size,
        )
    else:
        camera, rms = capture_calibration(
            args.output, args.camera, board_size=board_size,
            target_views=args.views, square_size=args.square_size,
        )
    print(f"Saved calibration to {args.output}; RMS reprojection error: {rms:.4f}px")
    print(camera.K)


if __name__ == "__main__":
    main()
