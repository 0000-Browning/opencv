from __future__ import annotations

import cv2
import numpy as np

from .feature_engine import FeatureEngine
from .models import Camera, Frame, KeyFrame, SlamMap, projection_matrix


def initialize_two_view(
    first: Frame,
    second: Frame,
    features: FeatureEngine,
    min_inliers: int = 30,
) -> tuple[SlamMap, list[tuple[int, int]]]:
    matches = features.match(first, second)
    if len(matches) < min_inliers:
        raise RuntimeError(f"Not enough geometrically verified matches ({len(matches)}; need {min_inliers}).")
    points0 = np.asarray([first.keypoints[match.queryIdx].pt for match in matches], dtype=np.float32)
    points1 = np.asarray([second.keypoints[match.trainIdx].pt for match in matches], dtype=np.float32)
    essential, mask = cv2.findEssentialMat(
        points0,
        points1,
        first.camera.K,
        method=cv2.USAC_MAGSAC,
        prob=0.999,
        threshold=1.0,
    )
    if essential is None or mask is None:
        raise RuntimeError("Essential matrix estimation failed; move the camera and try again.")
    _, rotation, translation, pose_mask = cv2.recoverPose(
        essential, points0, points1, first.camera.K, mask=mask
    )
    inlier_mask = pose_mask.reshape(-1).astype(bool)
    if int(inlier_mask.sum()) < min_inliers:
        raise RuntimeError(f"Two-view initialization has only {int(inlier_mask.sum())} cheirality inliers.")

    pose0 = np.eye(4, dtype=np.float64)
    pose1 = np.eye(4, dtype=np.float64)
    pose1[:3, :3] = rotation
    pose1[:3, 3] = translation.reshape(3)
    keyframe0 = KeyFrame.from_frame(first)
    keyframe1 = KeyFrame.from_frame(second)
    keyframe0.pose = pose0
    keyframe1.pose = pose1
    slam_map = SlamMap()
    slam_map.add_keyframe(keyframe0)
    slam_map.add_keyframe(keyframe1)

    selected_matches = [match for match, keep in zip(matches, inlier_mask) if keep]
    selected0 = np.asarray(
        [first.keypoints[match.queryIdx].pt for match in selected_matches], dtype=np.float64
    ).T
    selected1 = np.asarray(
        [second.keypoints[match.trainIdx].pt for match in selected_matches], dtype=np.float64
    ).T
    homogeneous = cv2.triangulatePoints(
        projection_matrix(first.camera, pose0),
        projection_matrix(first.camera, pose1),
        selected0,
        selected1,
    )
    valid_homogeneous = np.abs(homogeneous[3]) > 1e-10
    homogeneous[:, valid_homogeneous] /= homogeneous[3, valid_homogeneous]
    points3d = homogeneous[:3].T
    projected1 = (rotation @ points3d.T + translation.reshape(3, 1))[2]
    for index, match in enumerate(selected_matches):
        if not valid_homogeneous[index] or points3d[index, 2] <= 0 or projected1[index] <= 0:
            continue
        point = slam_map.add_point(points3d[index], first.descriptors[match.queryIdx])
        point.observations = {keyframe0.id: match.queryIdx, keyframe1.id: match.trainIdx}
        keyframe0.map_points[match.queryIdx] = point
        keyframe1.map_points[match.trainIdx] = point
    if len(slam_map.points) < min_inliers:
        raise RuntimeError(f"Only {len(slam_map.points)} points survived triangulation; improve baseline/scene texture.")
    return slam_map, [(match.queryIdx, match.trainIdx) for match in selected_matches]


def collect_initial_frames(camera: Camera, camera_index: int = 0) -> tuple[Frame, Frame]:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}.")
    features = FeatureEngine()
    captures: list[Frame] = []
    frame_id = 0
    try:
        while len(captures) < 2:
            success, image = capture.read()
            if not success:
                raise RuntimeError("Failed to read video while waiting for initialization.")
            if image is None:
                raise RuntimeError("Camera returned an empty frame during initialization.")
            frame = features.extract(Frame(frame_id, np.asarray(image, dtype=np.uint8), camera))
            frame_id += 1
            display = image.copy()
            status = "Aim at textured scene; press SPACE to capture keyframe 1"
            if captures:
                status = "Translate camera 10-20cm; press SPACE to capture keyframe 2"
            cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(display, "Q: quit", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Monocular SLAM initialization", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise RuntimeError("Initialization cancelled.")
            if key == ord(" ") and len(frame.keypoints) >= 30:
                captures.append(frame)
    finally:
        capture.release()
        cv2.destroyWindow("Monocular SLAM initialization")
    return captures[0], captures[1]
