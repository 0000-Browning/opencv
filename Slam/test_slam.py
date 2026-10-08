import cv2
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .feature_engine import FeatureEngine
from .initializer import initialize_two_view
from .models import Camera, Frame, KeyFrame, MapPoint, SlamMap, camera_center
from .optimizer import should_insert_keyframe
from .scale_aligner import scale_from_keyframe_clicks
from .tracker import Tracker
from .visualization import draw_persistent_features
from .visualize_ply import load_ascii_ply


class FixedMatches(FeatureEngine):
    def __init__(self, count: int) -> None:
        super().__init__()
        self.count = count

    def match(self, _first: Frame, _second: Frame) -> list[cv2.DMatch]:
        if _first.descriptors is None or _second.descriptors is None:
            return []
        return [cv2.DMatch(index, index, 0.0) for index in range(self.count)]


class SlamCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = Camera(np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]), np.zeros(5))

    def test_camera_center_for_world_to_camera_pose(self) -> None:
        pose = np.eye(4)
        pose[:3, 3] = (-1.0, 2.0, -3.0)
        np.testing.assert_allclose(camera_center(pose), (1.0, -2.0, 3.0))

    def test_prediction_advances_constant_translation(self) -> None:
        previous = np.eye(4)
        previous[0, 3] = 2.0
        before = np.eye(4)
        before[0, 3] = 1.0
        predicted = Tracker.predict_pose(previous, before)
        self.assertAlmostEqual(predicted[0, 3], 3.0)

    def test_keyframe_criteria(self) -> None:
        self.assertTrue(should_insert_keyframe(69, 100, 0, 0))
        self.assertTrue(should_insert_keyframe(100, 100, 26, 0))
        self.assertTrue(should_insert_keyframe(100, 100, 0, 31))
        self.assertFalse(should_insert_keyframe(100, 100, 10, 20))

    def test_point_cloud_export_writes_ply(self) -> None:
        slam_map = SlamMap()
        slam_map.add_point(np.array((1.0, 2.0, 3.0)), np.ones(128, dtype=np.float32))
        with TemporaryDirectory() as directory:
            output = Path(directory) / "map.ply"
            slam_map.save_ply(output)
            contents = output.read_text(encoding="ascii")
        self.assertIn("element vertex 1", contents)
        self.assertTrue(contents.endswith("1 2 3\n"))

    def test_ascii_ply_loader_reads_exported_map(self) -> None:
        slam_map = SlamMap()
        slam_map.add_point(np.array((1.0, 2.0, 3.0)), np.ones(128, dtype=np.float32))
        with TemporaryDirectory() as directory:
            output = Path(directory) / "map.ply"
            slam_map.save_ply(output)
            points = load_ascii_ply(output)
        np.testing.assert_allclose(points, [[1.0, 2.0, 3.0]])

    def test_persistent_feature_overlay_filters_and_caps_landmarks(self) -> None:
        frame = Frame(0, np.zeros((80, 80, 3), dtype=np.uint8), self.camera)
        frame.keypoints = [
            cv2.KeyPoint(10.0, 10.0, 1.0),
            cv2.KeyPoint(30.0, 30.0, 1.0),
            cv2.KeyPoint(50.0, 50.0, 1.0),
        ]
        frame.map_points = []
        for observation_count in (1, 3, 2):
            point = MapPoint(
                len(frame.map_points),
                np.array((0.0, 0.0, 1.0)),
                np.ones(128, dtype=np.float32),
                observations={index: index for index in range(observation_count)},
            )
            frame.map_points.append(point)

        overlay, drawn_count = draw_persistent_features(frame, max_features=1)

        self.assertEqual(drawn_count, 1)
        self.assertGreater(int(overlay[30, 30].sum()), 0)
        self.assertEqual(int(overlay[50, 50].sum()), 0)

    def test_scale_aligns_world_points_and_pose_translations(self) -> None:
        slam_map = SlamMap()
        frame = Frame(0, np.zeros((100, 100), dtype=np.uint8), self.camera)
        frame.keypoints = []
        frame.map_points = []
        keyframe = KeyFrame.from_frame(frame)
        keyframe.pose[0, 3] = 1.0
        first = slam_map.add_point(np.array((0.0, 0.0, 2.0)), np.ones(128, dtype=np.float32))
        second = slam_map.add_point(np.array((2.0, 0.0, 2.0)), np.ones(128, dtype=np.float32))
        keyframe.keypoints = [
            cv2.KeyPoint(10.0, 10.0, 1.0),
            cv2.KeyPoint(20.0, 10.0, 1.0),
        ]
        keyframe.map_points = [first, second]
        slam_map.add_keyframe(keyframe)
        scale = scale_from_keyframe_clicks(slam_map, keyframe, (10, 10), (20, 10), 0.3)
        self.assertAlmostEqual(scale, 0.15)
        np.testing.assert_allclose(second.pos - first.pos, (0.3, 0.0, 0.0))
        self.assertAlmostEqual(keyframe.pose[0, 3], 0.15)

    def test_two_view_initialization_triangulates_positive_depth_points(self) -> None:
        rng = np.random.default_rng(42)
        world_points = np.column_stack((
            rng.uniform(-1.0, 1.0, 80),
            rng.uniform(-0.7, 0.7, 80),
            rng.uniform(4.0, 10.0, 80),
        ))
        first = Frame(0, np.zeros((480, 640), dtype=np.uint8), self.camera)
        second = Frame(1, np.zeros((480, 640), dtype=np.uint8), self.camera)
        first.keypoints = []
        second.keypoints = []
        first.descriptors = rng.random((80, 128), dtype=np.float32)
        second.descriptors = first.descriptors.copy()
        first.map_points = [None] * 80
        second.map_points = [None] * 80
        for point in world_points:
            first_pixel = self.camera.K @ point
            second_pixel = self.camera.K @ (point + np.array((-0.5, 0.0, 0.0)))
            first.keypoints.append(cv2.KeyPoint(float(first_pixel[0] / first_pixel[2]), float(first_pixel[1] / first_pixel[2]), 1.0))
            second.keypoints.append(cv2.KeyPoint(float(second_pixel[0] / second_pixel[2]), float(second_pixel[1] / second_pixel[2]), 1.0))

        slam_map, matches = initialize_two_view(first, second, FixedMatches(80), min_inliers=30)

        self.assertEqual(len(matches), 80)
        self.assertGreaterEqual(len(slam_map.points), 30)
        self.assertTrue(all(point.pos[2] > 0 for point in slam_map.points.values()))

    def test_tracker_estimates_pose_from_synthetic_map_matches(self) -> None:
        rng = np.random.default_rng(7)
        slam_map = SlamMap()
        points = np.column_stack((
            rng.uniform(-1.0, 1.0, 40),
            rng.uniform(-0.7, 0.7, 40),
            rng.uniform(4.0, 8.0, 40),
        ))
        descriptors = rng.random((40, 128), dtype=np.float32)
        frame = Frame(2, np.zeros((480, 640), dtype=np.uint8), self.camera)
        frame.keypoints = []
        frame.descriptors = descriptors.copy()
        frame.map_points = [None] * 40
        true_translation = np.array((0.1, -0.05, 0.08))
        for index, point_position in enumerate(points):
            slam_map.add_point(point_position, descriptors[index])
            camera_point = point_position + true_translation
            pixel = self.camera.K @ camera_point
            frame.keypoints.append(
                cv2.KeyPoint(float(pixel[0] / pixel[2]), float(pixel[1] / pixel[2]), 1.0)
            )

        inliers = Tracker(FeatureEngine()).track(frame, slam_map, np.eye(4), None)

        self.assertGreaterEqual(inliers, 10)
        np.testing.assert_allclose(frame.pose[:3, 3], true_translation, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
