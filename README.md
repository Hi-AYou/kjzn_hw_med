# 任务2：场景目标检测与视频多目标跟踪

**作者**：易晨 &nbsp;|&nbsp; **学号**：25210980127 &nbsp;|&nbsp; **课程**：深度学习与空间智能（复旦大学，2026春）

---

## 任务说明

1. 使用 Road Vehicle Images Dataset 微调训练 YOLOv8m 目标检测模型
2. 对测试视频进行多目标跟踪（ByteTrack），为每个目标分配稳定的 Track ID
3. 分析遮挡场景下的 ID 维持与跳变现象
4. 实现虚拟线越线计数

---

## 目录结构

```
project/
├── configs/
│   └── dataset.yaml          # 数据集路径与类别配置
├── scripts/
│   ├── train.py              # 训练脚本
│   ├── track_and_count.py    # 多目标跟踪与越线计数
│   └── occlusion_analysis.py # 遮挡帧拼图可视化
└── outputs/                  # 训练/推理结果（运行后自动生成）
```

---

## 环境配置

Python 版本：3.8 及以上

```bash
pip install ultralytics>=8.4 opencv-python torch torchvision
```

可选实验追踪工具（二选一）：

```bash
pip install swanlab   # SwanLab
pip install wandb     # Weights & Biases
```

---

## 数据集

数据集来源：[Road Vehicle Images Dataset（Kaggle）](https://www.kaggle.com/datasets/ashfakyeafi/road-vehicle-images-dataset)

使用 kagglehub 下载：

```python
import kagglehub
path = kagglehub.dataset_download("ashfakyeafi/road-vehicle-images-dataset")
print("数据集路径：", path)
```

也可直接下载后解压，保证以下目录结构：

```
trafic_data/
├── train/
│   ├── images/   （2704 张）
│   └── labels/   （YOLO 格式 .txt）
├── valid/
│   ├── images/   （300 张）
│   └── labels/
└── data_1.yaml
```

然后修改 `configs/dataset.yaml` 中的 `path` 字段为实际路径。

---

## 训练

```bash
python scripts/train.py \
  --data   configs/dataset.yaml \
  --model  yolov8m.pt \
  --epochs 50 \
  --imgsz  640 \
  --batch  32 \
  --lr0    0.01 \
  --lrf    0.01 \
  --optimizer SGD \
  --device 0 \
  --name   yolov8m_road_vehicle
```

使用实验追踪（可选，二选一）：

```bash
python scripts/train.py ... --swanlab
python scripts/train.py ... --wandb
```

训练完成后权重保存在：

```
outputs/train/yolov8m_road_vehicle/weights/best.pt
outputs/train/yolov8m_road_vehicle/weights/last.pt
```

**主要训练参数：**

| 参数 | 值 |
|---|---|
| 模型 | YOLOv8m |
| 输入分辨率 | 640×640 |
| Batch Size | 32 |
| Epochs | 50 |
| 优化器 | SGD（momentum=0.937） |
| 初始学习率 | 0.01（余弦退火至 1e-4） |
| Warmup | 3 epochs |
| 数据增强 | Mosaic、MixUp、HSV 抖动、水平翻转 |

---

## 推理与跟踪

```bash
python scripts/track_and_count.py \
  --weights outputs/train/yolov8m_road_vehicle/weights/best.pt \
  --video   path/to/test_video.mov \
  --output  outputs/tracking \
  --tracker bytetrack.yaml \
  --conf    0.35 \
  --iou     0.50 \
  --device  0 \
  --line    0.1 0.6 0.9 0.6
```

参数说明：

| 参数 | 说明 |
|---|---|
| `--line X1 Y1 X2 Y2` | 计数线端点（相对坐标 0~1）。侧面拍摄用竖线如 `0.5 0.1 0.5 0.9`；面向行驶方向拍摄用横线如 `0.1 0.6 0.9 0.6` |
| `--occ-frames S E` | 指定保存遮挡分析截图的帧范围，如 `--occ-frames 200 204` |
| `--tracker` | 可选 `bytetrack.yaml` 或 `botsort.yaml` |

输出结果：

```
outputs/tracking/
├── tracked_output.mp4      # 带标注的输出视频
├── crossing_log.json       # 越线事件记录
└── occlusion_frames/       # 指定帧范围的截图
```

---

## 遮挡分析拼图

```bash
python scripts/occlusion_analysis.py \
  --frames-dir outputs/tracking/occlusion_frames \
  --output     outputs/tracking/occlusion_analysis.jpg \
  --width      800
```

---

## 模型权重与推理视频

预训练权重（best.pt）及推理视频已上传至 Google Drive：

[https://drive.google.com/drive/folders/1dtCoRMXjnihJspvMq1KXHtBRF2k6ZG1X](https://drive.google.com/drive/folders/1dtCoRMXjnihJspvMq1KXHtBRF2k6ZG1X)

包含文件：

- `best.pt`：YOLOv8m 微调最优权重（Epoch 40）
- `tracked_vertical.mp4`：竖线计数版本推理视频
- `tracked_horizontal.mp4`：横线计数版本推理视频

---

## 实验结果

在验证集（300 张）上的指标：

| mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|
| 0.560 | 0.329 | 0.658 | 0.538 |

推理速度：约 16.6 ms/张（Tesla T4），可满足实时处理需求。
