"""
视频多目标跟踪与越线计数
作者：易晨，学号：25210980127
"""

import sys
import json
import argparse
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

CLASS_NAMES = [
    "ambulance", "army vehicle", "auto rickshaw", "bicycle", "bus",
    "car", "garbagevan", "human hauler", "minibus", "minivan",
    "motorbike", "pickup", "policecar", "rickshaw", "scooter",
    "suv", "taxi", "three wheelers -CNG-", "truck", "van", "wheelbarrow"
]


def id_to_color(track_id: int):
    np.random.seed(int(track_id) % 2**16)
    return tuple(int(c) for c in np.random.randint(50, 230, size=3))


class LineCrossingCounter:
    """
    虚拟线越线计数器。
    利用向量叉积的符号判断目标中心点在线的哪一侧，
    符号改变时记录一次越线事件。
    """

    def __init__(self, pt1, pt2):
        self.pt1 = np.array(pt1, dtype=float)
        self.pt2 = np.array(pt2, dtype=float)
        self._last_side = {}
        self.count_ab = 0
        self.count_ba = 0

    def _side(self, point) -> float:
        d = self.pt2 - self.pt1
        return float(np.cross(d, np.array(point, dtype=float) - self.pt1))

    def update(self, track_id: int, cx: float, cy: float) -> int:
        current_side = self._side((cx, cy))
        if track_id not in self._last_side:
            self._last_side[track_id] = current_side
            return 0
        last = self._last_side[track_id]
        self._last_side[track_id] = current_side
        if last * current_side < 0:
            if last < 0:
                self.count_ab += 1
                return +1
            else:
                self.count_ba += 1
                return -1
        return 0

    @property
    def total(self):
        return self.count_ab + self.count_ba

    def draw(self, frame):
        p1 = tuple(int(v) for v in self.pt1)
        p2 = tuple(int(v) for v in self.pt2)
        cv2.line(frame, p1, p2, (0, 255, 255), 3)
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.putText(frame, f"Total crosses: {self.total}",
                    (mid[0] - 80, mid[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"A->B: {self.count_ab}  B->A: {self.count_ba}",
                    (mid[0] - 80, mid[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 0), 2)
        return frame


def run_tracking(
    weights: str,
    video_path: str,
    output_dir: str,
    tracker: str = "bytetrack.yaml",
    conf: float = 0.35,
    iou: float = 0.5,
    imgsz: int = 640,
    device: str = "",
    line_ratio: tuple = (0.1, 0.6, 0.9, 0.6),
    occlusion_frames: tuple = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "occlusion_frames"
    frames_dir.mkdir(exist_ok=True)

    model = YOLO(weights)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频：{video_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频分辨率：{W}x{H}，帧率：{fps:.1f}fps，总帧数：{total_frames}")

    x1r, y1r, x2r, y2r = line_ratio
    line_pt1 = (int(x1r * W), int(y1r * H))
    line_pt2 = (int(x2r * W), int(y2r * H))
    counter = LineCrossingCounter(line_pt1, line_pt2)

    out_video = str(output_dir / "tracked_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (W, H))

    track_history = defaultdict(list)
    crossing_log = []

    if occlusion_frames is None:
        mid = total_frames // 2
        occlusion_frames = (mid, mid + 4)
    occ_start, occ_end = occlusion_frames
    saved_occ_frames = []

    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker=tracker,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device if device else None,
            verbose=False,
        )

        annotated = frame.copy()

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes   = results[0].boxes.xyxy.cpu().numpy()
            ids     = results[0].boxes.id.int().cpu().numpy()
            classes = results[0].boxes.cls.int().cpu().numpy()
            confs   = results[0].boxes.conf.cpu().numpy()

            for box, tid, cls, cf in zip(boxes, ids, classes, confs):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                color = id_to_color(tid)
                label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                txt = f"#{tid} {label} {cf:.2f}"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(annotated, txt, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                track_history[tid].append((cx, cy))
                if len(track_history[tid]) > 40:
                    track_history[tid].pop(0)
                pts = np.array(track_history[tid], dtype=np.int32)
                cv2.polylines(annotated, [pts], False, color, 2)
                cv2.circle(annotated, (cx, cy), 4, color, -1)

                cross = counter.update(tid, cx, cy)
                if cross != 0:
                    direction = "A→B" if cross > 0 else "B→A"
                    crossing_log.append({
                        "frame": frame_idx, "track_id": int(tid),
                        "class": label, "direction": direction,
                        "cx": cx, "cy": cy,
                    })
                    flash_col = (0, 0, 255) if cross > 0 else (0, 255, 0)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), flash_col, 4)

        counter.draw(annotated)
        cv2.putText(annotated, f"Frame {frame_idx:04d}",
                    (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        if occ_start <= frame_idx < occ_end:
            saved_path = str(frames_dir / f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(saved_path, annotated)
            saved_occ_frames.append(saved_path)

        writer.write(annotated)
        frame_idx += 1

        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            print(f"  已处理 {frame_idx}/{total_frames} 帧  ({frame_idx/elapsed:.1f} fps)")

    cap.release()
    writer.release()
    elapsed = time.time() - t0
    print(f"\n完成：{frame_idx} 帧，耗时 {elapsed:.1f}s，平均 {frame_idx/elapsed:.1f} fps")
    print(f"输出视频：{out_video}")
    print(f"越线次数：{counter.total}（A→B: {counter.count_ab}，B→A: {counter.count_ba}）")

    log_path = output_dir / "crossing_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "crossings": crossing_log,
            "summary": {
                "total": counter.total,
                "A_to_B": counter.count_ab,
                "B_to_A": counter.count_ba,
            }
        }, f, indent=2, ensure_ascii=False)

    return {
        "video_out":        out_video,
        "crossing_log":     str(log_path),
        "occlusion_frames": saved_occ_frames,
        "total_crosses":    counter.total,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 多目标跟踪与越线计数")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--video",   type=str, required=True)
    parser.add_argument("--output",  type=str,
                        default=str(PROJECT_ROOT / "outputs" / "tracking"))
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml",
                        choices=["bytetrack.yaml", "botsort.yaml"])
    parser.add_argument("--conf",    type=float, default=0.35)
    parser.add_argument("--iou",     type=float, default=0.5)
    parser.add_argument("--imgsz",   type=int,   default=640)
    parser.add_argument("--device",  type=str,   default="")
    parser.add_argument("--line",    type=float, nargs=4,
                        default=[0.1, 0.6, 0.9, 0.6],
                        metavar=("X1R", "Y1R", "X2R", "Y2R"))
    parser.add_argument("--occ-frames", type=int, nargs=2, default=None,
                        metavar=("START", "END"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tracking(
        weights=args.weights,
        video_path=args.video,
        output_dir=args.output,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        line_ratio=tuple(args.line),
        occlusion_frames=tuple(args.occ_frames) if args.occ_frames else None,
    )
