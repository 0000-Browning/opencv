from __future__ import annotations

import cv2
import numpy as np

from .feature_engine import FeatureEngine
from .models import Frame, MapPoint, SlamMap


class Tracker:
    def __init__(self, features: FeatureEngine, minimum_inliers: int = 10) -> None:
        self.features = features
        self.minimum_inliers = minimum_inliers
        self.lost_frames = 0

    @staticmethod
    def predict_pose(previous: np.ndarray, before_previous: np.ndarray | None) -> np.ndarray:
        if before_previous is None:
            return previous.copy()
        relative_motion = previous @ np.linalg.inv(before_previous)
        return relative_motion @ previous

    def track(self, frame: Frame, slam_map: SlamMap, previous_pose: np.ndarray, before_previous: np.ndarray | None) -> int:
        points = list(slam_map.points.values())
        if frame.descriptors is None or len(points) < 4 or len(frame.descriptors) < 2:
            self.lost_frames += 1
            return 0
        map_descriptors = np.asarray([point.descriptor for point in points], dtype=np.float32)
        pairs = self.features.match_descriptors(map_descriptors, frame.descriptors)
        pairs.sort(key=lambda match: match.distance)
        used_keypoints: set[int] = set()
        correspondences: list[tuple[MapPoint, int]] = []
        for match in pairs:
            if match.trainIdx in used_keypoints:
                continue
            used_keypoints.add(match.trainIdx)
            correspondences.append((points[match.queryIdx], match.trainIdx))
        if len(correspondences) < self.minimum_inliers:
            self.lost_frames += 1
            return 0

        object_points = np.asarray([point.pos for point, _ in correspondences], dtype=np.float32)
        image_points = np.asarray(
            [frame.keypoints[index].pt for _, index in correspondences], dtype=np.float32
        )
        predicted = self.predict_pose(previous_pose, before_previous)
        rvec, _ = cv2.Rodrigues(predicted[:3, :3])
        tvec = predicted[:3, 3].reshape(3, 1).copy()
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            frame.camera.K,
            frame.camera.dist,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            iterationsCount=100,
            reprojectionError=2.0,
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success or inliers is None or len(inliers) < self.minimum_inliers:
            self.lost_frames += 1
            return 0
        inlier_indices = np.asarray(inliers).reshape(-1).astype(np.int32)
        if len(inlier_indices) >= 6:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points[inlier_indices],
                image_points[inlier_indices],
                frame.camera.K,
                frame.camera.dist,
                rvec,
                tvec,
            )
        rotation, _ = cv2.Rodrigues(rvec)
        frame.pose = np.eye(4, dtype=np.float64)
        frame.pose[:3, :3] = rotation
        frame.pose[:3, 3] = tvec.reshape(3)
        for index in inlier_indices:
            point, keypoint_index = correspondences[int(index)]
            frame.map_points[keypoint_index] = point
        self.lost_frames = 0
        return len(inlier_indices)
