#!/usr/bin/env python3
"""
YOLOv8 Object Detection Training Script
This script trains a YOLOv8 model on a custom object detection dataset.
Dataset should be in YOLO format with a data.yaml configuration file.
"""

import os
import sys
import yaml
import torch
from pathlib import Path
from datetime import datetime
import argparse
import logging
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YOLOv8Trainer:
    """
    A class to handle YOLOv8 model training, validation, and testing.
    """
    
    def __init__(self, data_yaml_path, model_name='yolov8n.pt', device='auto'):
        """
        Initialize the YOLOv8 trainer.
        
        Args:
            data_yaml_path (str): Path to the data.yaml file
            model_name (str): YOLOv8 model variant or path to custom weights
                            Options: 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 
                                   'yolov8l.pt', 'yolov8x.pt' or path to custom model
            device (str): Device to use for training ('cpu', 'cuda', 'mps', or 'auto')
        """
        self.data_yaml_path = Path(data_yaml_path)
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.model = None
        self.results = None
        
        # Verify data.yaml exists
        if not self.data_yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found at {self.data_yaml_path}")
        
        # Load and verify dataset configuration
        self._verify_dataset()
        
    def _setup_device(self, device):
        """
        Set up the training device (GPU/CPU).
        
        Args:
            device (str): Device preference
            
        Returns:
            str: Selected device
        """
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                device = 'mps'
                logger.info("Using Apple Silicon GPU (MPS)")
            else:
                device = 'cpu'
                logger.info("Using CPU for training")
        else:
            logger.info(f"Using specified device: {device}")
        
        return device
    
    def _verify_dataset(self):
        """
        Verify the dataset structure and configuration.
        """
        logger.info("Verifying dataset configuration...")
        
        # Load data.yaml
        with open(self.data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Check required fields
        required_fields = ['train', 'val', 'nc', 'names']
        for field in required_fields:
            if field not in data_config:
                raise ValueError(f"Missing required field '{field}' in data.yaml")
        
        # Verify paths exist
        base_path = self.data_yaml_path.parent
        train_path = base_path / data_config['train']
        val_path = base_path / data_config['val']
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found at {train_path}")
        if not val_path.exists():
            raise FileNotFoundError(f"Validation data not found at {val_path}")
        
        logger.info(f"Dataset verified: {data_config['nc']} classes")
        logger.info(f"Classes: {data_config['names']}")
    
    def initialize_model(self):
        """
        Initialize the YOLOv8 model.
        """
        logger.info(f"Initializing model: {self.model_name}")
        
        # Load model
        self.model = YOLO(self.model_name)
        
        # Log model information
        logger.info(f"Model loaded successfully")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.model.parameters()):,}")
    
    def train(self, 
              epochs=100,
              batch_size=16,
              imgsz=640,
              lr0=0.01,
              lrf=0.01,
              momentum=0.937,
              weight_decay=0.0005,
              warmup_epochs=3,
              warmup_momentum=0.8,
              warmup_bias_lr=0.1,
              box=7.5,
              cls=0.5,
              dfl=1.5,
              patience=50,
              save_period=10,
              workers=8,
              pretrained=True,
              optimizer='SGD',
              verbose=True,
              seed=42,
              deterministic=True,
              single_cls=False,
              rect=False,
              cos_lr=False,
              close_mosaic=10,
              resume=False,
              amp=True,
              fraction=1.0,
              profile=False,
              overlap_mask=True,
              mask_ratio=4,
              dropout=0.0,
              val=True,
              plots=True,
              save=True,
              save_json=False,
              save_hybrid=False,
              conf=None,
              iou=0.7,
              max_det=300,
              half=False,
              dnn=False,
              plots_format='png',
              cache=False,
              project='runs/detect',
              name='train',
              exist_ok=False):
        """
        Train the YOLOv8 model with specified hyperparameters.
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            imgsz (int): Input image size
            lr0 (float): Initial learning rate
            lrf (float): Final learning rate factor
            momentum (float): SGD momentum/Adam beta1
            weight_decay (float): Optimizer weight decay
            warmup_epochs (int): Warmup epochs
            warmup_momentum (float): Warmup initial momentum
            warmup_bias_lr (float): Warmup initial bias lr
            box (float): Box loss gain
            cls (float): Classification loss gain
            dfl (float): DFL loss gain
            patience (int): Early stopping patience
            save_period (int): Save checkpoint every x epochs
            workers (int): Number of worker threads for data loading
            pretrained (bool): Use pretrained weights
            optimizer (str): Optimizer to use ('SGD', 'Adam', 'AdamW', 'RMSProp')
            verbose (bool): Verbose output
            seed (int): Random seed for reproducibility
            deterministic (bool): Use deterministic algorithms
            single_cls (bool): Train as single-class dataset
            rect (bool): Rectangular training
            cos_lr (bool): Use cosine learning rate scheduler
            close_mosaic (int): Disable mosaic augmentation for final epochs
            resume (bool): Resume training from last checkpoint
            amp (bool): Automatic Mixed Precision training
            fraction (float): Dataset fraction to train on
            profile (bool): Profile ONNX and TensorRT speeds
            overlap_mask (bool): Masks overlap (instance segmentation)
            mask_ratio (int): Mask downsample ratio (segment train)
            dropout (float): Dropout rate
            val (bool): Validate during training
            plots (bool): Generate plots
            save (bool): Save train checkpoints
            save_json (bool): Save results to JSON
            save_hybrid (bool): Save hybrid version of model
            conf (float): Confidence threshold for predictions
            iou (float): IoU threshold for NMS
            max_det (int): Maximum detections per image
            half (bool): Use FP16 half-precision inference
            dnn (bool): Use OpenCV DNN for ONNX inference
            plots_format (str): Format for saving plots
            cache (bool): Cache images for faster training
            project (str): Project name
            name (str): Experiment name
            exist_ok (bool): Overwrite existing experiment
            
        Returns:
            dict: Training results
        """
        if self.model is None:
            self.initialize_model()
        
        logger.info("Starting training...")
        logger.info(f"Training parameters:")
        logger.info(f"  - Epochs: {epochs}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Image size: {imgsz}")
        logger.info(f"  - Initial LR: {lr0}")
        logger.info(f"  - Optimizer: {optimizer}")
        logger.info(f"  - Device: {self.device}")
        
        # Train the model
        self.results = self.model.train(
            data=str(self.data_yaml_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            warmup_momentum=warmup_momentum,
            warmup_bias_lr=warmup_bias_lr,
            box=box,
            cls=cls,
            dfl=dfl,
            patience=patience,
            save_period=save_period,
            workers=workers,
            pretrained=pretrained,
            optimizer=optimizer,
            verbose=verbose,
            seed=seed,
            deterministic=deterministic,
            single_cls=single_cls,
            rect=rect,
            cos_lr=cos_lr,
            close_mosaic=close_mosaic,
            resume=resume,
            amp=amp,
            fraction=fraction,
            profile=profile,
            overlap_mask=overlap_mask,
            mask_ratio=mask_ratio,
            dropout=dropout,
            val=val,
            plots=plots,
            save=save,
            save_json=save_json,
            save_hybrid=save_hybrid,
            conf=conf,
            iou=iou,
            max_det=max_det,
            half=half,
            dnn=dnn,
            device=self.device,
            project=project,
            name=name,
            exist_ok=exist_ok
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Best model saved at: {self.results.save_dir}/weights/best.pt")
        logger.info(f"Last model saved at: {self.results.save_dir}/weights/last.pt")
        
        return self.results
    
    def validate(self, model_path=None, conf=0.25, iou=0.7, max_det=300):
        """
        Validate the model on the validation dataset.
        
        Args:
            model_path (str): Path to model weights (uses best.pt from training if None)
            conf (float): Confidence threshold
            iou (float): IoU threshold for NMS
            max_det (int): Maximum detections per image
            
        Returns:
            dict: Validation metrics
        """
        if model_path is None and self.results is not None:
            model_path = f"{self.results.save_dir}/weights/best.pt"
        elif model_path is None:
            raise ValueError("No model path specified and no training results available")
        
        logger.info(f"Running validation with model: {model_path}")
        
        # Load model for validation
        val_model = YOLO(model_path)
        
        # Run validation
        metrics = val_model.val(
            data=str(self.data_yaml_path),
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=self.device
        )
        
        # Log validation results
        logger.info("Validation Results:")
        logger.info(f"  - mAP50: {metrics.box.map50:.4f}")
        logger.info(f"  - mAP50-95: {metrics.box.map:.4f}")
        logger.info(f"  - Precision: {metrics.box.mp:.4f}")
        logger.info(f"  - Recall: {metrics.box.mr:.4f}")
        
        return metrics
    
    def test(self, model_path=None, conf=0.25, iou=0.7, max_det=300):
        """
        Test the model on the test dataset.
        
        Args:
            model_path (str): Path to model weights
            conf (float): Confidence threshold
            iou (float): IoU threshold for NMS
            max_det (int): Maximum detections per image
            
        Returns:
            dict: Test metrics
        """
        if model_path is None and self.results is not None:
            model_path = f"{self.results.save_dir}/weights/best.pt"
        elif model_path is None:
            raise ValueError("No model path specified and no training results available")
        
        # Load data.yaml to check if test set exists
        with open(self.data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        if 'test' not in data_config:
            logger.warning("No test set defined in data.yaml")
            return None
        
        logger.info(f"Running testing with model: {model_path}")
        
        # Create temporary data.yaml with test set as val
        test_yaml = self.data_yaml_path.parent / 'test_data.yaml'
        data_config['val'] = data_config['test']
        with open(test_yaml, 'w') as f:
            yaml.dump(data_config, f)
        
        # Load model for testing
        test_model = YOLO(model_path)
        
        # Run test
        metrics = test_model.val(
            data=str(test_yaml),
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=self.device
        )
        
        # Clean up temporary file
        test_yaml.unlink()
        
        # Log test results
        logger.info("Test Results:")
        logger.info(f"  - mAP50: {metrics.box.map50:.4f}")
        logger.info(f"  - mAP50-95: {metrics.box.map:.4f}")
        logger.info(f"  - Precision: {metrics.box.mp:.4f}")
        logger.info(f"  - Recall: {metrics.box.mr:.4f}")
        
        return metrics
    
    def export_model(self, model_path=None, format='onnx', imgsz=640, half=False, dynamic=False, simplify=False):
        """
        Export the trained model to different formats.
        
        Args:
            model_path (str): Path to model weights
            format (str): Export format ('onnx', 'torchscript', 'coreml', 'tflite', etc.)
            imgsz (int): Image size for export
            half (bool): Use FP16 quantization
            dynamic (bool): ONNX/TensorRT dynamic axes
            simplify (bool): ONNX simplify
            
        Returns:
            str: Path to exported model
        """
        if model_path is None and self.results is not None:
            model_path = f"{self.results.save_dir}/weights/best.pt"
        elif model_path is None:
            raise ValueError("No model path specified and no training results available")
        
        logger.info(f"Exporting model to {format} format...")
        
        # Load model for export
        export_model = YOLO(model_path)
        
        # Export model
        export_path = export_model.export(
            format=format,
            imgsz=imgsz,
            half=half,
            dynamic=dynamic,
            simplify=simplify,
            device=self.device
        )
        
        logger.info(f"Model exported successfully to: {export_path}")
        return export_path


def main():
    """
    Main function to run the training pipeline.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train YOLOv8 on custom object detection dataset')
    
    # Dataset arguments
    parser.add_argument('--data', type=str, default='gun_detection_dataset/data.yaml',
                       help='Path to data.yaml file')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                       help='YOLOv8 model variant')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Initial learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use for training')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of worker threads for data loading')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience')
    
    # Validation/Test arguments
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for predictions')
    parser.add_argument('--iou', type=float, default=0.7,
                       help='IoU threshold for NMS')
    
    # Other arguments
    parser.add_argument('--no-train', action='store_true',
                       help='Skip training and only run validation/test')
    parser.add_argument('--no-val', action='store_true',
                       help='Skip validation')
    parser.add_argument('--no-test', action='store_true',
                       help='Skip testing')
    parser.add_argument('--export', type=str, default=None,
                       choices=['onnx', 'torchscript', 'coreml', 'tflite'],
                       help='Export format for the model')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from last checkpoint')
    parser.add_argument('--project', type=str, default='runs/detect',
                       help='Project directory for saving results')
    parser.add_argument('--name', type=str, default='train',
                       help='Experiment name')
    parser.add_argument('--exist-ok', action='store_true',
                       help='Overwrite existing experiment')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = YOLOv8Trainer(
            data_yaml_path=args.data,
            model_name=args.model,
            device=args.device
        )
        
        # Training phase
        if not args.no_train:
            results = trainer.train(
                epochs=args.epochs,
                batch_size=args.batch_size,
                imgsz=args.imgsz,
                lr0=args.lr,
                patience=args.patience,
                workers=args.workers,
                resume=args.resume,
                project=args.project,
                name=args.name,
                exist_ok=args.exist_ok,
                conf=args.conf,
                iou=args.iou
            )
        
        # Validation phase
        if not args.no_val:
            val_metrics = trainer.validate(
                conf=args.conf,
                iou=args.iou
            )
        
        # Test phase
        if not args.no_test:
            test_metrics = trainer.test(
                conf=args.conf,
                iou=args.iou
            )
        
        # Export model if requested
        if args.export:
            export_path = trainer.export_model(
                format=args.export,
                imgsz=args.imgsz
            )
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()