"""tests/test_detector.py — Unit tests for the detection engine."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def make_dummy_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestDetection:
    def test_detection_properties(self):
        from src.detector import Detection
        det = Detection(
            bbox=[100, 100, 300, 400],
            class_id=0,
            class_name="person",
            confidence=0.85,
        )
        assert det.width == 200
        assert det.height == 300
        assert det.area == 60000
        assert det.center == (200, 250)

    def test_detection_to_dict(self):
        from src.detector import Detection
        det = Detection(
            bbox=[10, 20, 110, 120],
            class_id=1,
            class_name="car",
            confidence=0.92,
            track_id=5,
        )
        d = det.to_dict()
        assert d["class_name"] == "car"
        assert d["track_id"] == 5
        assert d["confidence"] == 0.92


class TestDetectionResult:
    def test_class_counts(self):
        from src.detector import Detection, DetectionResult
        dets = [
            Detection([0, 0, 10, 10], 0, "person", 0.9),
            Detection([0, 0, 10, 10], 0, "person", 0.8),
            Detection([0, 0, 10, 10], 2, "car", 0.75),
        ]
        import time
        result = DetectionResult(dets, 12.5, frame_id=1, fps=30.0)
        assert result.count == 3
        assert result.class_counts() == {"person": 2, "car": 1}

    def test_by_class(self):
        from src.detector import Detection, DetectionResult
        import time
        dets = [
            Detection([0, 0, 10, 10], 0, "dog", 0.9),
            Detection([0, 0, 10, 10], 1, "cat", 0.7),
            Detection([0, 0, 10, 10], 0, "dog", 0.85),
        ]
        result = DetectionResult(dets, 8.0)
        by_cls = result.by_class()
        assert len(by_cls["dog"]) == 2
        assert len(by_cls["cat"]) == 1


class TestSpeedTracker:
    def test_tick_returns_float(self):
        from src.detector import SpeedTracker
        import time
        st = SpeedTracker()
        st.tick()
        time.sleep(0.05)
        fps = st.tick()
        assert isinstance(fps, float)

    def test_window_capping(self):
        from src.detector import SpeedTracker
        import time
        st = SpeedTracker(window=3)
        for _ in range(10):
            st.tick()
            time.sleep(0.01)
        assert len(st._times) <= 3


class TestObjectDetector:
    """Integration-style tests (require ultralytics to be installed)."""

    @pytest.fixture(scope="class")
    def detector(self):
        pytest.importorskip("ultralytics")
        from src.detector import ObjectDetector
        return ObjectDetector(model_name="yolo11n.pt", device="cpu", verbose=False)

    def test_predict_returns_result(self, detector):
        from src.detector import DetectionResult
        frame = make_dummy_frame()
        result = detector.predict(frame)
        assert isinstance(result, DetectionResult)
        assert result.inference_time_ms > 0

    def test_draw_returns_frame(self, detector):
        frame = make_dummy_frame()
        result = detector.predict(frame)
        annotated = detector.draw(frame, result)
        assert annotated.shape == frame.shape

    def test_class_names_populated(self, detector):
        assert len(detector.class_names) > 0

    def test_warmup(self, detector):
        detector.warmup()  # Should not raise
