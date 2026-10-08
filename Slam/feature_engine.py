from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from .models import Frame


class FeatureEngine:
    def __init__(self, nfeatures: int = 2000, ratio: float = 0.7) -> None:
        sift_factory = getattr(cv2, "SIFT_create", None)
        if sift_factory is None:
            raise RuntimeError("This OpenCV build does not provide SIFT support.")
        self.detector = sift_factory(
            nfeatures=nfeatures,
            contrastThreshold=0.04,
            edgeThreshold=10,
        )
        self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        self.ratio = ratio

    def extract(self, frame: Frame) -> Frame:
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY) if frame.image.ndim == 3 else frame.image
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        frame.keypoints = keypoints
        frame.descriptors = descriptors
        frame.map_points = [None] * len(keypoints)
        return frame

    def match(self, first: Frame, second: Frame) -> list[cv2.DMatch]:
        if first.descriptors is None or second.descriptors is None:
            return []
        if len(first.descriptors) < 2 or len(second.descriptors) < 2:
            return []
        pairs = self.matcher.knnMatch(first.descriptors, second.descriptors, k=2)
        good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < self.ratio * pair[1].distance]
        if len(good) < 8:
            return []
        points_first = np.asarray(
            [first.keypoints[item.queryIdx].pt for item in good], dtype=np.float32
        )
        points_second = np.asarray(
            [second.keypoints[item.trainIdx].pt for item in good], dtype=np.float32
        )
        fundamental, inlier_mask = cv2.findFundamentalMat(
            points_first, points_second, cv2.FM_RANSAC, 1.0, 0.99
        )
        if fundamental is None or inlier_mask is None:
            return []
        return [match for match, inlier in zip(good, inlier_mask.reshape(-1)) if inlier]

    def match_descriptors(
        self,
        query: NDArray[np.float32],
        train: NDArray[np.float32],
        ratio: float | None = None,
    ) -> list[cv2.DMatch]:
        if len(query) < 1 or len(train) < 2:
            return []
        pairs = self.matcher.knnMatch(query.astype(np.float32), train.astype(np.float32), k=2)
        cutoff = self.ratio if ratio is None else ratio
        return [
            pair[0]
            for pair in pairs
            if len(pair) == 2 and pair[0].distance < cutoff * pair[1].distance
        ]
