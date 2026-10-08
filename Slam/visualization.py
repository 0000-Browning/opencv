from __future__ import annotations

import cv2
import numpy as np

from .models import Frame, SlamMap, camera_center


def draw_persistent_features(
    frame: Frame,
    max_features: int = 120,
    min_keyframe_observations: int = 2,
) -> tuple[np.ndarray, int]:
    display = frame.image.copy()
    candidates = [
        (len(point.observations), point.id, keypoint.pt)
        for keypoint, point in zip(frame.keypoints, frame.map_points)
        if point is not None and len(point.observations) >= min_keyframe_observations
    ]
    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected = candidates[:max_features]
    for _, _, (x, y) in selected:
        location = (round(x), round(y))
        cv2.circle(display, location, 6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(display, location, 4, (0, 255, 80), 2, cv2.LINE_AA)
        cv2.circle(display, location, 1, (255, 255, 255), -1, cv2.LINE_AA)
    return display, len(selected)


def render_top_down(slam_map: SlamMap, width: int = 640, height: int = 480) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (24, 24, 24)
    if not slam_map.points and not slam_map.trajectory:
        cv2.putText(canvas, "Waiting for map", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)
        return canvas
    positions = [point.pos[[0, 2]] for point in slam_map.points.values()]
    positions.extend(camera_center(pose)[[0, 2]] for pose in slam_map.trajectory)
    coordinates = np.asarray(positions)
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    span = np.maximum(maximum - minimum, 0.1)
    scale = 0.85 * min((width - 40) / span[0], (height - 40) / span[1])

    def project(position: np.ndarray) -> tuple[int, int]:
        x = int((position[0] - minimum[0]) * scale + 20)
        y = int(height - 20 - (position[1] - minimum[1]) * scale)
        return x, y

    for point in slam_map.points.values():
        cv2.circle(canvas, project(point.pos[[0, 2]]), 2, (220, 180, 80), -1)
    trajectory = [project(camera_center(pose)[[0, 2]]) for pose in slam_map.trajectory]
    for start, end in zip(trajectory, trajectory[1:]):
        cv2.line(canvas, start, end, (40, 220, 40), 2)
    for position in trajectory:
        cv2.circle(canvas, position, 4, (40, 40, 240), -1)
    cv2.putText(
        canvas, f"Map points: {len(slam_map.points)} | Keyframes: {len(slam_map.keyframes)}",
        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1,
    )
    return canvas
