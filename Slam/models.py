from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass
class Camera:
    K: FloatArray
    dist: FloatArray

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=np.float64)
        self.dist = np.asarray(self.dist, dtype=np.float64).reshape(-1, 1)
        if self.K.shape != (3, 3) or not np.isfinite(self.K).all():
            raise ValueError("Camera intrinsics must be a finite 3x3 matrix.")
        if self.dist.size not in (4, 5, 8, 12, 14) or not np.isfinite(self.dist).all():
            raise ValueError("Distortion coefficients must contain 4, 5, 8, 12, or 14 values.")
        if abs(np.linalg.det(self.K)) < 1e-12:
            raise ValueError("Camera intrinsic matrix must be invertible.")

    @property
    def K_inv(self) -> FloatArray:
        return np.asarray(np.linalg.inv(self.K), dtype=np.float64)

    @classmethod
    def load(cls, path: str | Path) -> Camera:
        with Path(path).open(encoding="utf-8") as camera_file:
            data = json.load(camera_file)
        try:
            return cls(np.asarray(data["camera_matrix"]), np.asarray(data["distortion_coefficients"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid camera calibration file: {path}") from exc

    def save(self, path: str | Path, image_size: tuple[int, int], rms: float) -> None:
        output = {
            "camera_matrix": self.K.tolist(),
            "distortion_coefficients": self.dist.reshape(-1).tolist(),
            "image_size": [int(image_size[0]), int(image_size[1])],
            "rms_reprojection_error": float(rms),
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as camera_file:
            json.dump(output, camera_file, indent=2)


@dataclass
class Frame:
    id: int
    image: NDArray[np.uint8]
    camera: Camera
    timestamp: float = field(default_factory=time.time)
    keypoints: list[cv2.KeyPoint] = field(default_factory=list)
    descriptors: NDArray[np.float32] | None = None
    pose: FloatArray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    map_points: list[MapPoint | None] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.pose = np.asarray(self.pose, dtype=np.float64).reshape(4, 4)
        if not self.map_points and self.keypoints:
            self.map_points = [None] * len(self.keypoints)


@dataclass
class KeyFrame(Frame):
    @classmethod
    def from_frame(cls, frame: Frame) -> KeyFrame:
        return cls(
            id=frame.id,
            image=frame.image.copy(),
            camera=frame.camera,
            timestamp=frame.timestamp,
            keypoints=list(frame.keypoints),
            descriptors=None if frame.descriptors is None else frame.descriptors.copy(),
            pose=frame.pose.copy(),
            map_points=list(frame.map_points),
        )


@dataclass
class MapPoint:
    id: int
    pos: FloatArray
    descriptor: NDArray[np.float32]
    observations: dict[int, int] = field(default_factory=dict)
    normal: FloatArray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def __post_init__(self) -> None:
        self.pos = np.asarray(self.pos, dtype=np.float64).reshape(3)
        self.descriptor = np.asarray(self.descriptor, dtype=np.float32).reshape(-1)
        self.normal = np.asarray(self.normal, dtype=np.float64).reshape(3)


@dataclass
class SlamMap:
    keyframes: dict[int, KeyFrame] = field(default_factory=dict)
    points: dict[int, MapPoint] = field(default_factory=dict)
    trajectory: list[FloatArray] = field(default_factory=list)
    next_point_id: int = 0

    def add_point(self, position: FloatArray, descriptor: NDArray[np.float32]) -> MapPoint:
        point = MapPoint(self.next_point_id, position, descriptor)
        self.points[point.id] = point
        self.next_point_id += 1
        return point

    def add_keyframe(self, keyframe: KeyFrame) -> None:
        self.keyframes[keyframe.id] = keyframe
        self.trajectory.append(keyframe.pose.copy())

    def apply_scale(self, scale: float) -> None:
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Metric scale must be a finite positive number.")
        for point in self.points.values():
            point.pos *= scale
        for keyframe in self.keyframes.values():
            keyframe.pose[:3, 3] *= scale
            keyframe.pose[3, :] = (0.0, 0.0, 0.0, 1.0)
        self.trajectory = [pose.copy() for pose in (keyframe.pose for keyframe in self.keyframes.values())]

    def save_ply(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="ascii", newline="\n") as ply_file:
            ply_file.write(
                "ply\nformat ascii 1.0\n"
                f"element vertex {len(self.points)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "end_header\n"
            )
            for point in self.points.values():
                ply_file.write(f"{point.pos[0]:.9g} {point.pos[1]:.9g} {point.pos[2]:.9g}\n")


def projection_matrix(camera: Camera, world_to_camera: FloatArray) -> FloatArray:
    return camera.K @ np.asarray(world_to_camera, dtype=np.float64)[:3, :]


def camera_center(world_to_camera: FloatArray) -> FloatArray:
    pose = np.asarray(world_to_camera, dtype=np.float64).reshape(4, 4)
    return -pose[:3, :3].T @ pose[:3, 3]
