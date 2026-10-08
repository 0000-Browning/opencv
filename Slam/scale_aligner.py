from __future__ import annotations

import cv2
import numpy as np

from .models import KeyFrame, MapPoint, SlamMap


def scale_from_keyframe_clicks(
    slam_map: SlamMap,
    keyframe: KeyFrame,
    pixel0: tuple[int, int],
    pixel1: tuple[int, int],
    known_distance: float,
    max_pixel_distance: float = 15.0,
) -> float:
    if not np.isfinite(known_distance) or known_distance <= 0:
        raise ValueError("Known ruler distance must be a finite positive number.")
    if keyframe.id not in slam_map.keyframes:
        raise ValueError("Selected keyframe is not part of this map.")
    selected: list[MapPoint] = []
    for pixel in (pixel0, pixel1):
        candidates = [
            (float(np.linalg.norm(np.asarray(kp.pt) - np.asarray(pixel))), index, point)
            for index, (kp, point) in enumerate(zip(keyframe.keypoints, keyframe.map_points))
            if point is not None
        ]
        if not candidates:
            raise RuntimeError("Selected keyframe has no mapped feature points.")
        distance, _, point = min(candidates, key=lambda item: item[0])
        if distance > max_pixel_distance:
            raise RuntimeError(
                f"No mapped feature is within {max_pixel_distance:.0f}px of ruler click (nearest: {distance:.1f}px)."
            )
        selected.append(point)
    first_point, second_point = selected
    if first_point.id == second_point.id:
        raise ValueError("Ruler clicks selected the same reconstructed map point.")
    raw_distance = float(np.linalg.norm(first_point.pos - second_point.pos))
    if raw_distance < 1e-9:
        raise RuntimeError("Selected map points have a degenerate reconstructed distance.")
    scale = known_distance / raw_distance
    slam_map.apply_scale(scale)
    return scale


def select_ruler_points(
    slam_map: SlamMap,
    keyframe: KeyFrame,
    known_distance: float,
) -> float:
    clicks: list[tuple[int, int]] = []
    display = keyframe.image.copy()
    window_name = "Select ruler endpoints (left-click twice; Enter to confirm; Esc to cancel)"
    frozen = display.copy()

    def on_mouse(event: int, x: int, y: int, flags: int, data: object) -> None:
        del flags, data
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
            clicks.append((x, y))

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)
    try:
        while True:
            shown = frozen.copy()
            for point in clicks:
                cv2.circle(shown, point, 6, (0, 0, 255), -1)
            if len(clicks) == 2:
                cv2.line(shown, clicks[0], clicks[1], (0, 255, 0), 2)
            cv2.imshow(window_name, shown)
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                raise RuntimeError("Metric scale selection cancelled.")
            if key in (10, 13) and len(clicks) == 2:
                break
    finally:
        cv2.destroyWindow(window_name)
    return scale_from_keyframe_clicks(slam_map, keyframe, clicks[0], clicks[1], known_distance)
