# YOLOv8 Object Detection Training Script

## Introduction

This project aims to develop an advanced AI model for real-time gun detection in images captured from surveillance cameras. The broader vision is to enhance public safety by enabling automated systems to scan images for firearms and promptly alert security personnel, especially during high-profile events or gatherings where public figures are present and the risk of crime is elevated. By leveraging state-of-the-art object detection techniques, this solution seeks to provide reliable, fast, and accurate detection to help prevent incidents and improve response times in critical situations.

This repository provides a flexible and modular Python script to train YOLOv8 models on custom object detection datasets. It supports training, validation, testing, and exporting models in multiple formats (ONNX, TorchScript, CoreML, TFLite).

## Features
- Train YOLOv8 models from scratch or fine-tune pretrained weights
- Validate and test models with configurable confidence and IoU thresholds
- Supports automatic device selection (CPU, GPU, or Apple MPS)
- Export trained models to ONNX, TorchScript, CoreML, or TFLite
- Logging, progress tracking, and configurable hyperparameters
- Highly customizable training options: batch size, learning rate, image size, optimizer, augmentation, early stopping, and more

## Dataset Structure
The dataset should follow the YOLO format with the following structure:

```
dataset/
├─ data.yaml          # Dataset configuration
├─ train/
│  ├─ images/         # Training images
│  └─ labels/         # YOLO formatted labels
├─ val/
│  ├─ images/         # Validation images
│  └─ labels/         # YOLO formatted labels
└─ test/              # (optional)
   ├─ images/         # Test images (optional)
   └─ labels/         # YOLO formatted labels (optional)
```

The `data.yaml` file should include:
```yaml
train: path/to/train/images
val: path/to/val/images
test: path/to/test/images  # optional
nc: <number_of_classes>
names: [class1, class2, ...]
```

## Installation
1. Clone this repository:
   ```sh
   git clone <repository_url>
   cd <repository_folder>
   ```
2. Install dependencies:
   ```sh
   pip install ultralytics matplotlib pandas pyyaml
   ```
3. Verify your dataset structure and `data.yaml`

## Usage
Run the training script with default settings:
```sh
python train_yolov8.py --data path/to/data.yaml --model yolov8n.pt
```

### Common Arguments
- `--data`: Path to data.yaml file
- `--model`: YOLOv8 model variant (`yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`)
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size for training (default: 16)
- `--imgsz`: Input image size (default: 640)
- `--lr`: Initial learning rate (default: 0.01)
- `--device`: Device to use (`auto`, `cpu`, `cuda`, `mps`)
- `--workers`: Number of data loading workers (default: 8)
- `--conf`: Confidence threshold for predictions (default: 0.25)
- `--iou`: IoU threshold for NMS (default: 0.7)
- `--export`: Export format (`onnx`, `torchscript`, `coreml`, `tflite`)
- `--resume`: Resume training from last checkpoint

#### Additional Flags
- `--no-train`, `--no-val`, `--no-test`, `--exist-ok`

## Examples
Train a YOLOv8n model for 50 epochs:
```sh
python train_yolov8.py --data dataset/data.yaml --model yolov8n.pt --epochs 50
```
Train on GPU and export to ONNX:
```sh
python train_yolov8.py --data dataset/data.yaml --model yolov8s.pt --device cuda --export onnx
```
Skip training and only validate/test:
```sh
python train_yolov8.py --data dataset/data.yaml --no-train
```

## Outputs
Training results are saved in:
```
runs/detect/<experiment_name>/
├─ weights/
│  ├─ best.pt      # Best model checkpoint
│  └─ last.pt      # Last model checkpoint
├─ results.png     # Training curves and metrics plots
└─ ...             # Other training logs and plots
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.
