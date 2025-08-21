#!/usr/bin/env python3
"""
Script to download and setup ML models for FastAPI app
"""
import os
from huggingface_hub import snapshot_download
from ultralytics import YOLO
import logging
import glob
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_all_models():
    """Download all required models"""
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # 1. Move YOLO person detection model if it exists in root
    yolo_target = "models/yolo11s.pt"
    if not os.path.exists(yolo_target):
        if os.path.exists("yolo11s.pt"):
            logger.info("Moving yolo11s.pt from root to models/ ...")
            shutil.move("yolo11s.pt", yolo_target)
            logger.info("yolo11s.pt has been moved successfully!")
        else:
            logger.warning("File yolo11s.pt not found in root. Please place the file in the root folder.")
    else:
        logger.info("YOLO11s model already exists in models/.")
    
    # 2. Download shelf segmentation model
    if not os.path.exists("models/shelf_model/best.pt"):
        logger.info("Downloading shelf segmentation model...")
        snapshot_download(
            repo_id="cheesecz/shelf-segmentation", 
            local_dir="models/shelf_model", 
            local_dir_use_symlinks=False
        )
        logger.info("Shelf model downloaded!")
    
    # 3. Download action classification model
    config_file = "models/action_model/config.json"
    if not os.path.exists(config_file):
        logger.info("Downloading action classification model...")
        snapshot_download(
            repo_id="haipradana/tracko-videomae-action-detection",
            local_dir="models/action_model",
            local_dir_use_symlinks=False
        )
        logger.info("Action model downloaded!")
    
    logger.info("All models downloaded successfully!")

if __name__ == "__main__":
    download_all_models()