"""
train.py — Fine-tune any YOLO model on a custom dataset.

Usage:
  python train.py --data data/custom_dataset.yaml --model yolo11n.pt --epochs 100
  python train.py --data data/custom_dataset.yaml --model yolo11m.pt --batch 32 --device cuda
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune YOLO on custom data")
    p.add_argument("--data", required=True, help="Dataset YAML (Roboflow / custom)")
    p.add_argument("--model", default="yolo11n.pt",
                   help="Base model checkpoint")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="auto")
    p.add_argument("--lr0", type=float, default=0.001)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default="custom_model")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cache", action="store_true", help="Cache images in RAM")
    p.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    p.add_argument("--freeze", type=int, default=0,
                   help="Number of backbone layers to freeze")
    return p.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def main():
    args = parse_args()
    device = resolve_device(args.device)
    logger.info(f"Training on device: {device}")

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    model = YOLO(args.model)
    logger.info(f"Loaded base model: {args.model}")
    logger.info(f"Dataset: {args.data}")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        lr0=args.lr0,
        patience=args.patience,
        project=args.project,
        name=args.name,
        workers=args.workers,
        cache=args.cache,
        resume=args.resume,
        freeze=args.freeze if args.freeze > 0 else None,
        plots=True,
        save=True,
        val=True,
        verbose=True,
    )

    logger.info("Training complete!")
    best = Path(args.project) / args.name / "weights" / "best.pt"
    logger.info(f"Best model → {best}")
    logger.info(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    logger.info(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")

    # Validate best model
    logger.info("Running validation on best model...")
    best_model = YOLO(str(best))
    val_results = best_model.val(data=args.data, device=device)
    logger.info("Validation done.")


if __name__ == "__main__":
    main()
