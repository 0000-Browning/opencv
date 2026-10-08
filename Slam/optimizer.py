from __future__ import annotations

import importlib

import cv2
import numpy as np

from .feature_engine import FeatureEngine
from .models import KeyFrame, SlamMap, projection_matrix


def should_insert_keyframe(
    inlier_count: int,
    initial_map_point_count: int,
    median_displacement: float,
    frames_since_keyframe: int,
) -> bool:
    if initial_map_point_count <= 0:
        return True
    return (
        inlier_count < 0.7 * initial_map_point_count
        or median_displacement > 25.0
        or frames_since_keyframe > 30
    )


def triangulate_new_points(
    slam_map: SlamMap,
    current: KeyFrame,
    neighbor: KeyFrame,
    features: FeatureEngine,
    minimum_parallax_degrees: float = 0.5,
) -> int:
    if current.descriptors is None or neighbor.descriptors is None:
        return 0
    matches = features.match(current, neighbor)
    unmatched = [
        match for match in matches
        if current.map_points[match.queryIdx] is None and neighbor.map_points[match.trainIdx] is None
    ]
    if not unmatched:
        return 0
    pixels_current = np.asarray(
        [current.keypoints[m.queryIdx].pt for m in unmatched], dtype=np.float64
    ).T
    pixels_neighbor = np.asarray(
        [neighbor.keypoints[m.trainIdx].pt for m in unmatched], dtype=np.float64
    ).T
    homogeneous = cv2.triangulatePoints(
        projection_matrix(current.camera, current.pose),
        projection_matrix(neighbor.camera, neighbor.pose),
        pixels_current,
        pixels_neighbor,
    )
    centers = [
        -pose[:3, :3].T @ pose[:3, 3]
        for pose in (current.pose, neighbor.pose)
    ]
    added = 0
    for index, match in enumerate(unmatched):
        if abs(homogeneous[3, index]) < 1e-10:
            continue
        position = homogeneous[:3, index] / homogeneous[3, index]
        depths = [
            (pose[:3, :3] @ position + pose[:3, 3])[2]
            for pose in (current.pose, neighbor.pose)
        ]
        if min(depths) <= 0:
            continue
        ray0 = position - centers[0]
        ray1 = position - centers[1]
        cosine = np.clip(np.dot(ray0, ray1) / (np.linalg.norm(ray0) * np.linalg.norm(ray1)), -1, 1)
        parallax = np.degrees(np.arccos(cosine))
        if parallax < minimum_parallax_degrees:
            continue
        projected0 = current.camera.K @ (current.pose[:3, :3] @ position + current.pose[:3, 3])
        projected1 = neighbor.camera.K @ (neighbor.pose[:3, :3] @ position + neighbor.pose[:3, 3])
        error0 = np.linalg.norm(projected0[:2] / projected0[2] - pixels_current[:, index])
        error1 = np.linalg.norm(projected1[:2] / projected1[2] - pixels_neighbor[:, index])
        if max(error0, error1) > 3.0:
            continue
        point = slam_map.add_point(position, current.descriptors[match.queryIdx])
        point.observations = {current.id: match.queryIdx, neighbor.id: match.trainIdx}
        current.map_points[match.queryIdx] = point
        neighbor.map_points[match.trainIdx] = point
        added += 1
    return added


def bundle_adjustment(slam_map: SlamMap, window_size: int = 5) -> float:
    try:
        scipy_optimize = importlib.import_module("scipy.optimize")
        scipy_sparse = importlib.import_module("scipy.sparse")
    except ImportError as exc:
        raise RuntimeError(
            "Local bundle adjustment requires SciPy. Install scipy or run without --bundle-adjustment."
        ) from exc

    keyframes = list(slam_map.keyframes.values())[-window_size:]
    if len(keyframes) < 2:
        return 0.0
    point_ids = sorted({
        point.id
        for keyframe in keyframes
        for point in keyframe.map_points
        if point is not None and len(point.observations) >= 2
    })
    if len(point_ids) < 6:
        return 0.0
    point_index = {point_id: i for i, point_id in enumerate(point_ids)}
    fixed_keyframe_id = keyframes[0].id
    variable_keyframes = [kf for kf in keyframes if kf.id != fixed_keyframe_id]
    pose_index = {kf.id: i for i, kf in enumerate(variable_keyframes)}
    observations: list[tuple[int, int, np.ndarray]] = []
    for keyframe in keyframes:
        for point_id, keypoint_index in (
            (point.id, index) for index, point in enumerate(keyframe.map_points) if point is not None
        ):
            if point_id in point_index:
                observations.append((keyframe.id, point_id, np.asarray(keyframe.keypoints[keypoint_index].pt)))
    if len(observations) < 12:
        return 0.0

    pose_values: list[float] = []
    for keyframe in variable_keyframes:
        rotation_vector, _ = cv2.Rodrigues(keyframe.pose[:3, :3])
        pose_values.extend(rotation_vector.reshape(3))
        pose_values.extend(keyframe.pose[:3, 3])
    initial = np.concatenate((np.asarray(pose_values), np.asarray([
        coordinate for point_id in point_ids for coordinate in slam_map.points[point_id].pos
    ])))
    n_poses, n_points = len(variable_keyframes), len(point_ids)

    def unpack(parameters: np.ndarray) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], np.ndarray]:
        poses: dict[int, tuple[np.ndarray, np.ndarray]] = {
            keyframes[0].id: (keyframes[0].pose[:3, :3], keyframes[0].pose[:3, 3])
        }
        for keyframe in variable_keyframes:
            offset = pose_index[keyframe.id] * 6
            rotation, _ = cv2.Rodrigues(parameters[offset : offset + 3])
            poses[keyframe.id] = (rotation, parameters[offset + 3 : offset + 6])
        points_start = n_poses * 6
        points = parameters[points_start:].reshape(n_points, 3)
        return poses, points

    def residuals(parameters: np.ndarray) -> np.ndarray:
        poses, points = unpack(parameters)
        result = np.empty((len(observations), 2), dtype=np.float64)
        for row, (keyframe_id, point_id, observed) in enumerate(observations):
            rotation, translation = poses[keyframe_id]
            camera_point = rotation @ points[point_index[point_id]] + translation
            if camera_point[2] <= 1e-8:
                result[row] = 1e3
            else:
                pixel = keyframes[0].camera.K @ camera_point
                result[row] = pixel[:2] / pixel[2] - observed
        return result.ravel()

    sparsity = scipy_sparse.lil_matrix(
        (2 * len(observations), 6 * n_poses + 3 * n_points), dtype=np.int8
    )
    for row, (keyframe_id, point_id, _) in enumerate(observations):
        if keyframe_id in pose_index:
            start = pose_index[keyframe_id] * 6
            sparsity[2 * row : 2 * row + 2, start : start + 6] = 1
        start = 6 * n_poses + 3 * point_index[point_id]
        sparsity[2 * row : 2 * row + 2, start : start + 3] = 1
    optimized = scipy_optimize.least_squares(
        residuals, initial, jac_sparsity=sparsity.tocsr(), loss="huber", f_scale=2.0,
        max_nfev=20, verbose=0,
    )
    poses, points = unpack(optimized.x)
    for keyframe in variable_keyframes:
        keyframe.pose[:3, :3], keyframe.pose[:3, 3] = poses[keyframe.id]
    for point_id, index in point_index.items():
        slam_map.points[point_id].pos = points[index].copy()
    slam_map.trajectory = [kf.pose.copy() for kf in slam_map.keyframes.values()]
    return float(np.mean(np.abs(optimized.fun)))
