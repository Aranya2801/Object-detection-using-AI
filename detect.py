"""
detect.py — Universal inference CLI
Run object detection on webcam, video, images, or RTSP streams.

Usage examples:
  python detect.py --source 0                          # webcam
  python detect.py --source video.mp4 --track
  python detect.py --source image.jpg --model yolo11m.pt
  python detect.py --source rtsp://... --model yolo11n.pt --conf 0.5
  python detect.py --source images/ --save
"""

import argparse
import cv2
import json
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from src.detector import ObjectDetector
from utils.io import open_source, is_image, save_frame, VideoWriter
from utils.zone import ZoneCounter
from utils.analytics import AnalyticsLogger


def parse_args():
    p = argparse.ArgumentParser(
        description="Advanced Object Detection — YOLO11 / YOLOv8 / Custom Models",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Source
    p.add_argument("--source", default="0",
                   help="Input: 0=webcam, path to image/video/folder, RTSP URL")
    # Model
    p.add_argument("--model", default="yolo11n.pt",
                   help="Model: yolo11n/s/m/l/x, yolov8n/s/m/l/x, or custom .pt path")
    p.add_argument("--task", default="detect",
                   choices=["detect", "segment", "pose", "classify", "obb"],
                   help="Inference task")
    # Thresholds
    p.add_argument("--conf", type=float, default=0.40, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--max-det", type=int, default=300, help="Max detections per frame")
    p.add_argument("--classes", nargs="+", type=int, default=None,
                   help="Filter by class IDs (e.g. --classes 0 2 5)")
    # Device
    p.add_argument("--device", default="auto",
                   help="Device: auto, cpu, cuda, cuda:0, mps")
    # Tracking
    p.add_argument("--track", action="store_true", help="Enable multi-object tracking")
    p.add_argument("--tracker", default="bytetrack.yaml",
                   choices=["bytetrack.yaml", "botsort.yaml"],
                   help="Tracker algorithm")
    # Display / Output
    p.add_argument("--show", action="store_true", default=True,
                   help="Display output window")
    p.add_argument("--no-show", dest="show", action="store_false")
    p.add_argument("--save", action="store_true", help="Save output to data/outputs/")
    p.add_argument("--save-json", action="store_true",
                   help="Save per-frame detection JSON")
    p.add_argument("--save-csv", action="store_true",
                   help="Append detection stats to CSV log")
    # Drawing
    p.add_argument("--no-labels", action="store_true", help="Hide class labels")
    p.add_argument("--no-conf", action="store_true", help="Hide confidence scores")
    p.add_argument("--no-mask", action="store_true", help="Hide segmentation masks")
    p.add_argument("--thickness", type=int, default=2, help="Bounding box line thickness")
    # Zone
    p.add_argument("--zone", action="store_true",
                   help="Enable interactive zone/ROI counting")
    # Misc
    p.add_argument("--warmup", action="store_true", help="Warm up GPU before inference")
    p.add_argument("--fps-limit", type=int, default=0,
                   help="Cap processing FPS (0 = unlimited)")
    p.add_argument("--verbose", action="store_true", help="Verbose YOLO output")
    return p.parse_args()


def run(args):
    # ── Detector ────────────────────────────────────────────────────────────
    detector = ObjectDetector(
        model_name=args.model,
        task=args.task,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        max_det=args.max_det,
        classes=args.classes,
        track=args.track,
        tracker=args.tracker,
        verbose=args.verbose,
    )
    if args.warmup:
        detector.warmup()

    # ── Output dir ──────────────────────────────────────────────────────────
    out_dir = Path("data/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Analytics logger ────────────────────────────────────────────────────
    analytics = AnalyticsLogger(out_dir) if (args.save_json or args.save_csv) else None

    # ── Zone counter ────────────────────────────────────────────────────────
    zone = ZoneCounter() if args.zone else None

    # ── Source ──────────────────────────────────────────────────────────────
    source_str = args.source
    try:
        source_int = int(source_str)
        source_str = source_int  # webcam index
    except ValueError:
        pass

    if is_image(source_str):
        _run_image(detector, source_str, args, out_dir, analytics, zone)
    else:
        _run_video(detector, source_str, args, out_dir, analytics, zone)


def _run_image(detector, source, args, out_dir, analytics, zone):
    frame = cv2.imread(str(source))
    if frame is None:
        logger.error(f"Cannot read image: {source}")
        return
    result = detector.predict(frame)
    annotated = detector.draw(
        frame, result,
        show_labels=not args.no_labels,
        show_conf=not args.no_conf,
        show_mask=not args.no_mask,
        line_thickness=args.thickness,
    )
    logger.info(f"Detected {result.count} objects | {result.inference_time_ms:.1f} ms")
    logger.info(f"Counts: {result.class_counts()}")

    if args.save:
        out_path = out_dir / f"result_{Path(str(source)).stem}.jpg"
        cv2.imwrite(str(out_path), annotated)
        logger.info(f"Saved → {out_path}")
    if analytics:
        analytics.log(result)
    if args.show:
        cv2.imshow("Detection Result", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def _run_video(detector, source, args, out_dir, analytics, zone):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open source: {source}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if args.save:
        out_path = out_dir / "output.mp4"
        writer = VideoWriter(str(out_path), w, h, int(src_fps))
        logger.info(f"Saving video → {out_path}")

    frame_delay = 1 / args.fps_limit if args.fps_limit > 0 else 0
    logger.info("Press Q or ESC to quit.")

    try:
        while True:
            t_frame = time.time()
            ok, frame = cap.read()
            if not ok:
                break

            result = detector.predict(frame)

            if zone:
                zone.update(result.detections)

            annotated = detector.draw(
                frame, result,
                show_labels=not args.no_labels,
                show_conf=not args.no_conf,
                show_mask=not args.no_mask,
                line_thickness=args.thickness,
            )
            if zone:
                zone.draw(annotated)

            if args.show:
                cv2.imshow("Object Detection — Press Q to quit", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            if writer:
                writer.write(annotated)
            if analytics:
                analytics.log(result)

            if frame_delay > 0:
                elapsed = time.time() - t_frame
                remaining = frame_delay - elapsed
                if remaining > 0:
                    time.sleep(remaining)

    finally:
        cap.release()
        if writer:
            writer.release()
        if analytics:
            analytics.finalize()
        cv2.destroyAllWindows()
        logger.info("Done.")


if __name__ == "__main__":
    args = parse_args()
    run(args)
