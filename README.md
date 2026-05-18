# ComputerVision Midterm

本仓库为计算机视觉期中项目代码，包含三项实验：

1. **宠物识别**：在 Oxford-IIIT Pet Dataset 上微调 ImageNet 预训练 ResNet，并进行随机初始化和 SE 注意力消融。
2. **目标检测与多目标跟踪**：在 VisDrone2019-DET 上微调 YOLOv8，并在测试视频中输出 Bounding Box、类别、Tracking ID、遮挡片段和越线计数。
3. **语义分割**：从零手写 U-Net，在 Oxford-IIIT Pet 三分类分割任务上比较 CE、Dice、CE+Dice 三种损失。

实验报告：`midterm_report.pdf`  
课堂展示：`midterm_presentation.pdf`  
模型权重下载：<https://drive.google.com/drive/folders/1J0FnH9rXQwcwxuRReBn3vQhCzPGZpMQM>

## Environment

实验环境：

- Python 3.13.5
- PyTorch 2.11.0 + CUDA 12.8
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Main packages: `torch`, `torchvision`, `ultralytics`, `opencv-python`, `wandb`

安装依赖：

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

验证 CUDA：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Dataset Layout

数据集不上传到 GitHub，请按如下结构放置。

### Oxford-IIIT Pet

`torchvision.datasets.OxfordIIITPet` 默认读取：

```text
data/
  oxford-iiit-pet/
    annotations/
    images/
```

如果原目录权限异常，可使用本项目中的干净解压目录：

```text
data_clean/
  oxford-iiit-pet/
    annotations/
    images/
```

本实验 task1 实际使用 `data_clean`。

### VisDrone2019-DET

当前 `visdrone.yaml` 对应如下目录结构：

```text
data/VisDrone/
  VisDrone2019-DET-train/
    VisDrone2019-DET-train/
      images/
      annotations/
      labels/
  VisDrone2019-DET-val/
    VisDrone2019-DET-val/
      images/
      annotations/
      labels/
  VisDrone2019-DET-test-dev/
    images/
    annotations/
    labels/
```

如果重新解压 VisDrone，可运行：

```powershell
python task2_yolo_visdrone.py prepare --visdrone-root data/VisDrone --yaml visdrone.yaml
```

## Task 1: Pet Classification

主脚本：

```text
task1_pet_classification.py
```

ResNet-34 ImageNet 预训练 baseline：

```powershell
python task1_pet_classification.py --data-root data_clean --pretrained --arch resnet34 --epochs 50 --batch-size 32 --lr-head 1e-3 --lr-backbone 1e-4
```

随机初始化消融：

```powershell
python task1_pet_classification.py --data-root data_clean --random-init --arch resnet34 --epochs 10 --batch-size 32 --lr-head 1e-3 --lr-backbone 1e-3
```

SE 注意力模型：

```powershell
python task1_pet_classification.py --data-root data_clean --pretrained --arch resnet34 --attention se --epochs 10 --batch-size 32 --lr-head 1e-3 --lr-backbone 1e-4
```

结果：

| Model | Pretrained | Attention | Epoch | Best Accuracy |
|---|---:|---:|---:|---:|
| ResNet-34 | No | None | 10 | 0.2499 |
| ResNet-34 | Yes | None | 50 | 0.8910 |
| ResNet-34 | Yes | SE | 10 | 0.8918 |

输出文件：

```text
runs/task1/*.csv
runs/task1/*.pt
```

## Task 2: VisDrone Detection and Tracking

主脚本：

```text
task2_yolo_visdrone.py
```

训练 YOLOv8n：

```powershell
python task2_yolo_visdrone.py train --yaml visdrone.yaml --model yolov8n.pt --epochs 50 --imgsz 640 --batch 8
```

当前训练权重：

```text
runs/detect/runs/task2/visdrone_yolov8n-2/weights/best.pt
```

视频跟踪、遮挡帧截取与越线计数：

```powershell
python task2_yolo_visdrone.py track --weights runs/detect/runs/task2/visdrone_yolov8n-2/weights/best.pt --video data/test_video_3.mp4 --line 100 360 1180 360 --occlusion-start 120 --occlusion-frames 4
```

检测结果：

| Model | Epoch | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| YOLOv8n | 50 | 0.4316 | 0.3136 | 0.2924 | 0.1635 |

跟踪输出：

```text
runs/task2/track/tracked_counted.mp4
runs/task2/track/occlusion_000120.jpg
runs/task2/track/occlusion_000121.jpg
runs/task2/track/occlusion_000122.jpg
runs/task2/track/occlusion_000123.jpg
runs/task2/track/tracking_summary.json
```

当前测试视频越线计数结果为 `0`。

## Task 3: U-Net Segmentation

主脚本：

```text
task3_unet_segmentation.py
```

三种损失训练命令：

```powershell
python task3_unet_segmentation.py --loss ce --epochs 20 --batch-size 8 --size 128
python task3_unet_segmentation.py --loss dice --epochs 20 --batch-size 8 --size 128
python task3_unet_segmentation.py --loss ce_dice --epochs 20 --batch-size 8 --size 128
```

结果：

| Loss | Epoch | Best mIoU |
|---|---:|---:|
| Cross-Entropy | 20 | 0.7461 |
| Dice Loss | 20 | 0.7527 |
| Cross-Entropy + Dice Loss | 20 | 0.7563 |

输出文件：

```text
runs/task3/*.csv
runs/task3/*.pt
```

## WandB Visualization

复现 wandb 曲线可直接上传已有 CSV，无需重新训练：

```powershell
python -m pip install wandb
python -m wandb login
python log_existing_results_to_wandb.py --task task1 --max-epoch 10
python log_existing_results_to_wandb.py --task task2
python log_existing_results_to_wandb.py --task task3
```

对应项目：

```text
cv-midterm-task1
cv-midterm-task2
cv-midterm-task3
```

报告中使用的截图路径：

```text
figures/task1_wandb.png
figures/task2_wandb.png
figures/task3_wandb.png
```

## Result Summary

汇总本地 CSV 指标：

```powershell
python summarize_results.py
```

当前输出：

```text
Task 1 classification:
  resnet34_pre0_none_eh10_lh0.001_lb0.001.csv: best_val_acc=0.2499
  resnet34_pre1_none_eh50_lh0.001_lb0.0001.csv: best_val_acc=0.8910
  resnet34_pre1_se_eh10_lh0.001_lb0.0001.csv: best_val_acc=0.8918
Task 3 segmentation:
  unet_ce.csv: best_val_miou=0.7461
  unet_ce_dice.csv: best_val_miou=0.7563
  unet_dice.csv: best_val_miou=0.7527
```


runs/
*.pt
*.mp4
```
