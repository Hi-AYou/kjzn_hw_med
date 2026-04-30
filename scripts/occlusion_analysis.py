"""
遮挡分析帧可视化脚本
作者：易晨，学号：25210980127
"""

import sys
import argparse
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def make_grid(frame_paths: list, output_path: str, width: int = 800) -> str:
    imgs = []
    for p in frame_paths:
        img = cv2.imread(p)
        if img is None:
            print(f"  无法读取：{p}")
            continue
        imgs.append(img)

    if not imgs:
        raise FileNotFoundError("未找到有效帧图片。")

    h0, w0 = imgs[0].shape[:2]
    new_w = width
    new_h = int(h0 * width / w0)
    resized = [cv2.resize(im, (new_w, new_h)) for im in imgs]

    n = len(resized)
    if n <= 2:
        grid = np.hstack(resized)
    else:
        row1 = np.hstack(resized[:2])
        row2_imgs = resized[2:4]
        if len(row2_imgs) < 2:
            row2_imgs.append(np.zeros_like(resized[0]))
        row2 = np.hstack(row2_imgs)
        grid = np.vstack([row1, row2])

    cv2.imwrite(output_path, grid)
    print(f"  拼图已保存：{output_path}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="遮挡分析帧拼图")
    parser.add_argument("--frames-dir", type=str, required=True)
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "outputs" / "tracking" / "occlusion_analysis.jpg"))
    parser.add_argument("--width", type=int, default=800)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    frame_paths = sorted(frames_dir.glob("*.jpg"))[:4]
    if not frame_paths:
        print(f"未找到帧图片：{frames_dir}")
        raise SystemExit(1)
    make_grid([str(p) for p in frame_paths], args.output, args.width)
