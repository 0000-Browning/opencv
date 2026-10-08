from __future__ import annotations

import logging
import queue
import threading
import time
from importlib.util import find_spec
from dataclasses import dataclass

import cv2
import numpy as np

from .feature_engine import FeatureEngine
from .models import Camera, Frame, KeyFrame, SlamMap
from .optimizer import bundle_adjustment, should_insert_keyframe, triangulate_new_points
from .tracker import Tracker
from .visualization import draw_persistent_features, render_top_down

LOGGER = logging.getLogger(__name__)


@dataclass
class RenderUpdate:
    image: np.ndarray


class SlamSystem:
    def __init__(
        self,
        camera: Camera,
        slam_map: SlamMap,
        camera_index: int = 0,
        enable_bundle_adjustment: bool = False,
    ) -> None:
        if enable_bundle_adjustment and find_spec("scipy") is None:
            raise RuntimeError(
                "Local bundle adjustment was requested, but SciPy is not installed. "
                "Install scipy or disable --bundle-adjustment."
            )
        self.camera = camera
        self.map = slam_map
        self.camera_index = camera_index
        self.enable_bundle_adjustment = enable_bundle_adjustment
        self.map_lock = threading.RLock()
        self.keyframe_queue: queue.Queue[KeyFrame | None] = queue.Queue(maxsize=3)
        self.render_queue: queue.Queue[RenderUpdate] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.worker_error: Exception | None = None
        self.features = FeatureEngine()
        self.tracker = Tracker(self.features)
        self.frame_id = max(self.map.keyframes, default=-1) + 1
        self.last_keyframe_id = max(self.map.keyframes, default=-1)
        self.previous_pose = next(reversed(self.map.keyframes.values())).pose.copy()
        keyframes = list(self.map.keyframes.values())
        self.before_previous_pose = keyframes[-2].pose.copy() if len(keyframes) > 1 else None

    def _report_worker_error(self, error: Exception) -> None:
        self.worker_error = error
        LOGGER.exception("SLAM background worker failed", exc_info=error)
        self.stop_event.set()

    def mapping_loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                try:
                    current = self.keyframe_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if current is None:
                    self.keyframe_queue.task_done()
                    return
                with self.map_lock:
                    neighbors = list(self.map.keyframes.values())[-2:]
                    self.map.add_keyframe(current)
                    for neighbor in neighbors:
                        triangulate_new_points(self.map, current, neighbor, self.features)
                    if self.enable_bundle_adjustment:
                        error = bundle_adjustment(self.map, window_size=5)
                        LOGGER.info("Local BA mean absolute residual: %.3f px", error)
                    self.last_keyframe_id = current.id
                self.keyframe_queue.task_done()
        except Exception as exc:
            self._report_worker_error(exc)

    def visualization_loop(self) -> None:
        try:
            while not self.stop_event.wait(0.1):
                with self.map_lock:
                    image = render_top_down(self.map)
                update = RenderUpdate(image)
                try:
                    self.render_queue.put_nowait(update)
                except queue.Full:
                    try:
                        self.render_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.render_queue.put_nowait(update)
        except Exception as exc:
            self._report_worker_error(exc)

    def _queue_keyframe(self, frame: Frame) -> bool:
        keyframe = KeyFrame.from_frame(frame)
        try:
            self.keyframe_queue.put_nowait(keyframe)
            return True
        except queue.Full:
            LOGGER.warning("Mapping queue full; delaying keyframe insertion for frame %d.", frame.id)
            return False

    def _check_worker(self) -> None:
        if self.worker_error is not None:
            raise RuntimeError("A SLAM background worker failed.") from self.worker_error

    def run(self) -> None:
        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_index}.")
        mapping_thread = threading.Thread(target=self.mapping_loop, name="slam-mapping", daemon=True)
        visualization_thread = threading.Thread(target=self.visualization_loop, name="slam-visualization", daemon=True)
        mapping_thread.start()
        visualization_thread.start()
        timings: list[float] = []
        frames_since_keyframe = 0
        try:
            while not self.stop_event.is_set():
                self._check_worker()
                started = time.perf_counter()
                success, image = capture.read()
                if not success:
                    raise RuntimeError("Camera frame capture failed.")
                frame = self.features.extract(
                    Frame(self.frame_id, np.asarray(image, dtype=np.uint8), self.camera)
                )
                self.frame_id += 1
                if len(frame.keypoints) < 30:
                    status = "Low feature environment"
                    inliers = 0
                else:
                    with self.map_lock:
                        inliers = self.tracker.track(
                            frame, self.map, self.previous_pose, self.before_previous_pose
                        )
                    if inliers == 0 and self.tracker.lost_frames > 5:
                        status = "Tracking lost - move to a mapped view"
                    elif inliers == 0:
                        frame.pose = self.tracker.predict_pose(self.previous_pose, self.before_previous_pose)
                        status = f"Tracking fallback {self.tracker.lost_frames}/5"
                        if self.tracker.lost_frames <= 5:
                            self.before_previous_pose, self.previous_pose = self.previous_pose, frame.pose.copy()
                    else:
                        status = f"Tracking: {inliers} inliers"
                        self.before_previous_pose, self.previous_pose = self.previous_pose, frame.pose.copy()
                        frames_since_keyframe += 1
                        with self.map_lock:
                            map_count = len(self.map.points)
                            keyframe = self.map.keyframes.get(self.last_keyframe_id)
                            matched_pixels = [
                                np.linalg.norm(
                                    np.asarray(frame.keypoints[index].pt)
                                    - np.asarray(keyframe.keypoints[point.observations[self.last_keyframe_id]].pt)
                                )
                                for index, point in enumerate(frame.map_points)
                                if point is not None
                                and keyframe is not None
                                and self.last_keyframe_id in point.observations
                            ]
                        median_displacement = float(np.median(matched_pixels)) if matched_pixels else 0.0
                        if should_insert_keyframe(
                            inliers, map_count, median_displacement, frames_since_keyframe
                        ):
                            if self._queue_keyframe(frame):
                                frames_since_keyframe = 0
                with self.map_lock:
                    display, persistent_count = draw_persistent_features(frame)
                cv2.putText(display, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(
                    display, f"Features: {len(frame.keypoints)} | FPS: {1.0 / max(time.perf_counter() - started, 1e-6):.1f}",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
                )
                cv2.putText(
                    display, f"Persistent landmarks: {persistent_count}",
                    (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 1,
                )
                cv2.imshow("Monocular SLAM", display)
                try:
                    update = self.render_queue.get_nowait()
                    cv2.imshow("SLAM top-down map", update.image)
                except queue.Empty:
                    pass
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elapsed = time.perf_counter() - started
                timings.append(elapsed)
                if len(timings) >= 100:
                    LOGGER.info("Mean frame time over %d frames: %.2fms", len(timings), 1000 * np.mean(timings))
                    timings.clear()
        finally:
            self.stop_event.set()
            try:
                self.keyframe_queue.put_nowait(None)
            except queue.Full:
                pass
            capture.release()
            cv2.destroyAllWindows()
            mapping_thread.join(timeout=2.0)
            visualization_thread.join(timeout=2.0)
        self._check_worker()
