"""
YOLOv8 微调训练脚本
数据集：Road Vehicle Images Dataset
作者：易晨，学号：25210980127
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 训练脚本")
    parser.add_argument("--data",      type=str,
                        default=str(PROJECT_ROOT / "configs" / "dataset.yaml"))
    parser.add_argument("--model",     type=str, default="yolov8m.pt")
    parser.add_argument("--epochs",    type=int, default=50)
    parser.add_argument("--imgsz",     type=int, default=640)
    parser.add_argument("--batch",     type=int, default=16)
    parser.add_argument("--lr0",       type=float, default=0.01)
    parser.add_argument("--lrf",       type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="SGD",
                        choices=["SGD", "Adam", "AdamW", "auto"])
    parser.add_argument("--workers",   type=int, default=4)
    parser.add_argument("--project",   type=str,
                        default=str(PROJECT_ROOT / "outputs" / "train"))
    parser.add_argument("--name",      type=str, default="yolov8m_road_vehicle")
    parser.add_argument("--device",    type=str, default="")
    parser.add_argument("--resume",    action="store_true")
    parser.add_argument("--swanlab",   action="store_true")
    parser.add_argument("--wandb",     action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.swanlab:
        try:
            import swanlab
            swanlab.init(
                project="road-vehicle-detection",
                experiment_name=args.name,
                config=vars(args),
            )
        except ImportError:
            print("[Warning] swanlab 未安装，跳过。")

    if args.wandb:
        try:
            import wandb
            wandb.init(
                project="road-vehicle-detection",
                name=args.name,
                config=vars(args),
            )
        except ImportError:
            print("[Warning] wandb 未安装，跳过。")

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        lrf=args.lrf,
        optimizer=args.optimizer,
        workers=args.workers,
        project=args.project,
        name=args.name,
        device=args.device if args.device else None,
        resume=args.resume,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        weight_decay=0.0005,
        warmup_epochs=3,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    if best_weights.exists():
        best_model = YOLO(str(best_weights))
        metrics = best_model.val(data=args.data, imgsz=args.imgsz, split="val")
        print(f"mAP@0.5:      {metrics.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
        print(f"Precision:    {metrics.box.mp:.4f}")
        print(f"Recall:       {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
