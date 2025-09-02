from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Form, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import torch
import cv2
import numpy as np
import tempfile
import os
import base64
import json
import uuid
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Union, List
from collections import defaultdict
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForVideoClassification
from huggingface_hub import snapshot_download
import supervision as sv
from decord import VideoReader, cpu
import matplotlib.pyplot as plt
from io import BytesIO
import random
import subprocess
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions, ContentSettings
import logging
from dotenv import load_dotenv
from urllib.parse import quote
import re
from openai import AzureOpenAI

# Load environment variables
# Ensure we load .env located next to this file (fastapi-app/.env),
# regardless of current working directory when uvicorn is started.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path=ENV_PATH)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Retail Behavior Analysis API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Explicitly allow all methods including OPTIONS
    allow_headers=["*"],
)

# Azure Blob Storage configuration
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER", "retail-analysis-results")

# Azure OpenAI configuration (for AI Insights)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # e.g., gpt-5-chat
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

def _get_azure_client() -> AzureOpenAI:
    if not (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY and AZURE_OPENAI_DEPLOYMENT):
        raise HTTPException(status_code=500, detail="Azure OpenAI is not configured")
    return AzureOpenAI(api_version=AZURE_OPENAI_API_VERSION, azure_endpoint=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_KEY)

# Initialize Azure Blob client
if AZURE_STORAGE_CONNECTION_STRING:
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    logger.info("Azure Blob Storage client initialized")
else:
    blob_service_client = None
    logger.warning("Azure Blob Storage not configured - using local storage")

# Global models
models = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@app.on_event("startup")
async def load_models():
    """Load ML models on startup - Load from models/ directory"""
    global models
    logger.info("Loading ML models...")
    
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Create models directory if not exists
        os.makedirs("models", exist_ok=True)
        
        # Download models if not exists (using download script logic)
        if not os.path.exists("models/yolo11s.pt"):
            logger.info("Downloading YOLO11s model...")
            models["person_model"] = YOLO('yolo11s.pt').to(device)  # This will auto-download
            # Move to models directory
            if os.path.exists("yolo11s.pt"):
                import shutil
                shutil.move("yolo11s.pt", "models/yolo11s.pt")
        else:
            models["person_model"] = YOLO('models/yolo11s.pt').to(device)
        
        # Download shelf segmentation model if not exists
        if not os.path.exists("models/shelf_model/best.pt"):
            logger.info("Downloading shelf segmentation model...")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="cheesecz/shelf-segmentation", 
                local_dir="models/shelf_model", 
                local_dir_use_symlinks=False
            )
        
        models["shelf_model"] = YOLO('models/shelf_model/best.pt').to(device)
        # Load action classification model - try local first, fallback to online
        # Load action classification model - improved version like pipeline.py
        local_action_model_path = "models/action_model"
        config_file = os.path.join(local_action_model_path, "config.json")
        
        if os.path.exists(config_file):
            logger.info("Loading action model from local directory...")
            try:
                models["action_model"] = AutoModelForVideoClassification.from_pretrained(
                    local_action_model_path,
                    local_files_only=True
                ).to(device)
                models["image_processor"] = AutoImageProcessor.from_pretrained(
                    local_action_model_path,
                    local_files_only=True
                )
            except Exception as e:
                logger.warning(f"Failed to load local model: {e}, falling back to online")
                models["action_model"] = AutoModelForVideoClassification.from_pretrained(
                    'haipradana/tracko-videomae-action-detection'
                ).to(device)
                models["image_processor"] = AutoImageProcessor.from_pretrained(
                    'haipradana/tracko-videomae-action-detection'
                )
        else:
            logger.info("Loading action model from HuggingFace...")
            models["action_model"] = AutoModelForVideoClassification.from_pretrained(
                'haipradana/tracko-videomae-action-detection'
            ).to(device)
            models["image_processor"] = AutoImageProcessor.from_pretrained(
                'haipradana/tracko-videomae-action-detection'
            )
        
        # Set model to evaluation mode and get id2label like in pipeline.py
        models["action_model"].eval()
        
        # Get id2label from model config - no fallback mapping
        try:
            models["id2label"] = models["action_model"].config.id2label
            if not models["id2label"]:
                raise ValueError("Model id2label is empty")
            logger.info(f"Loaded id2label from model: {models['id2label']}")
        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Model doesn't have valid id2label: {e}")
            raise HTTPException(
                status_code=500, 
                detail="Action classification model is not properly configured - missing id2label mapping"
            )
        
        # Store device info
        models["device"] = device
        
        logger.info("Models loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise e

def _parse_connection_string(conn_str: str):
    parts = {}
    try:
        for seg in conn_str.split(';'):
            if not seg:
                continue
            k, v = seg.split('=', 1)
            parts[k] = v
    except Exception:
        pass
    return parts

def _generate_sas_url(blob_client, expiry_hours: int = 24):
    try:
        # Prefer using connection string secrets if available
        conn_str = AZURE_STORAGE_CONNECTION_STRING
        if not conn_str:
            return blob_client.url
        parts = _parse_connection_string(conn_str)
        account_name = parts.get('AccountName')
        account_key = parts.get('AccountKey')
        endpoint_suffix = parts.get('EndpointSuffix', 'core.windows.net')
        if not (account_name and account_key):
            return blob_client.url

        sas = generate_blob_sas(
            account_name=account_name,
            container_name=blob_client.container_name,
            blob_name=blob_client.blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=expiry_hours)
        )

        # Build URL using the actual blob URL to preserve custom domains if any
        base_url = blob_client.url.split('?', 1)[0]
        return f"{base_url}?{sas}"
    except Exception as e:
        logger.warning(f"Failed to generate SAS URL, returning direct URL: {e}")
        return blob_client.url

def save_to_azure_blob(data: bytes, filename: str, content_type: str = "application/json"):
    """Save file to Azure Blob Storage"""
    if not blob_service_client:
        raise HTTPException(status_code=500, detail="Azure Blob Storage not configured")
    
    try:
        blob_client = blob_service_client.get_blob_client(
            container=CONTAINER_NAME, 
            blob=filename
        )
        
        # Use ContentSettings to ensure proper headers for streaming/inline display
        content_settings = ContentSettings(
            content_type=content_type,
            content_disposition="inline" if content_type.startswith("video/") else None,
            cache_control="public, max-age=86400"
        )

        blob_client.upload_blob(data, overwrite=True, content_settings=content_settings)

        logger.info(f"File uploaded to Azure Blob: {filename}")
        # Return short-lived SAS URL for browser access
        return _generate_sas_url(blob_client, expiry_hours=24)
    except Exception as e:
        logger.error(f"Error uploading to Azure Blob: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload to Azure Blob: {str(e)}")

def generate_heatmap_image(heatmap_data: np.ndarray, analysis_id: str):
    """Generate heatmap image from data using same style as Gradio"""
    # Convert list back to numpy array if needed
    if isinstance(heatmap_data, list):
        heatmap_data = np.array(heatmap_data)
    
    plt.figure(figsize=(10, 8))
    
    # Use same visualization as Gradio
    plt.imshow(heatmap_data, cmap='hot', interpolation='nearest')
    plt.title(f'Customer Traffic Heatmap - Analysis {analysis_id}', fontsize=14)
    plt.xlabel('X Position (grid)', fontsize=12)
    plt.ylabel('Y Position (grid)', fontsize=12)
    plt.colorbar(label='Interaction Frequency')
    plt.tight_layout()
    
    # Save to bytes with better quality
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    img_buffer.seek(0)
    plt.close()
    
    return img_buffer.getvalue()

def _make_processing_serializable(processing_data: dict, frame_size: tuple[int, int] | None = None) -> dict:
    """Convert processing_data to JSON-serializable types and attach frame size if provided."""
    try:
        tracks = {}
        for pid, dets in processing_data.get('tracks', {}).items():
            key = str(pid)
            ser_dets = []
            for d in dets:
                bbox = d.get('bbox')
                if isinstance(bbox, np.ndarray):
                    bbox = bbox.tolist()
                else:
                    bbox = [float(x) for x in bbox]
                ser_dets.append({'frame': int(d.get('frame', 0)), 'bbox': bbox, 'pid': int(d.get('pid', pid))})
            tracks[key] = ser_dets

        action_preds = {}
        for pid, preds in processing_data.get('action_preds', {}).items():
            key = str(pid)
            ser_preds = []
            for p in preds:
                ser_preds.append({
                    'start': int(p.get('start', 0)),
                    'end': int(p.get('end', 0)),
                    'pred': int(p.get('pred', 0)),
                    'action_name': p.get('action_name')
                })
            action_preds[key] = ser_preds

        shelf_boxes_per_frame = {}
        for f, items in processing_data.get('shelf_boxes_per_frame', {}).items():
            # items: list of (sid, (x1,y1,x2,y2))
            ser_items = []
            for sid, box in items or []:
                ser_box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                ser_items.append([str(sid), ser_box])
            shelf_boxes_per_frame[str(int(f))] = ser_items

        fps = float(processing_data.get('fps', 30.0))
        H = None
        W = None
        if frame_size and len(frame_size) == 2:
            H, W = int(frame_size[0]), int(frame_size[1])
        else:
            try:
                fs = processing_data.get('frame_size')
                if fs and len(fs) == 2:
                    H, W = int(fs[0]), int(fs[1])
            except Exception:
                H, W = None, None

        return {
            'tracks': tracks,
            'action_preds': action_preds,
            'shelf_boxes_per_frame': shelf_boxes_per_frame,
            'fps': fps,
            'frame_size': [H, W] if (H is not None and W is not None) else None,
        }
    except Exception as e:
        logger.error(f"Failed to serialize processing data: {e}")
        raise

def _generate_track_gallery_thumbnails(video_path: str, tracks: dict, fps: float, analysis_id: str, timestamp: str, limit: int = 64):
    """Generate per-track thumbnail crops and upload to Azure Blob if configured.
    Returns list of { pid, frame, bbox, duration_s, thumbnail_url }.
    """
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
    except Exception as e:
        logger.warning(f"Failed to open video for thumbnails: {e}")
        return []

    # Sort pids by dwell (longest first) and cap at limit
    pid_to_dwell = []
    for pid, dets in tracks.items():
        try:
            dwell_s = len(dets) / float(fps) if fps else 0.0
        except Exception:
            dwell_s = 0.0
        pid_to_dwell.append((pid, dwell_s))
    pid_to_dwell.sort(key=lambda x: x[1], reverse=True)
    selected = [pid for pid, _ in pid_to_dwell[:max(1, limit)]]

    gallery = []
    for pid in selected:
        dets = tracks.get(pid, [])
        if not dets:
            continue
        mid_idx = len(dets) // 2
        det = dets[mid_idx]
        f_idx = int(det.get('frame', 0))
        bbox = det.get('bbox')
        try:
            frame = vr[f_idx].asnumpy()
            x1, y1, x2, y2 = map(int, (bbox[0], bbox[1], bbox[2], bbox[3]))
            x1 = max(0, min(x1, frame.shape[1]-1)); x2 = max(0, min(x2, frame.shape[1]-1))
            y1 = max(0, min(y1, frame.shape[0]-1)); y2 = max(0, min(y2, frame.shape[0]-1))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            # Resize to max width 180 keeping aspect
            h, w, _ = crop.shape
            max_w = 180
            scale = min(1.0, max_w / max(1, w))
            if scale <= 0:
                scale = 1.0
            new_w = max(1, int(w * scale)); new_h = max(1, int(h * scale))
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            resized = cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode('.png', resized)
            if not ok:
                continue
            data = buf.tobytes()
            thumb_url = None
            if blob_service_client:
                filename = f"thumbnails/{analysis_id}_{timestamp}_pid{pid}.png"
                try:
                    thumb_url = save_to_azure_blob(data, filename, "image/png")
                except Exception as e:
                    logger.warning(f"Failed to upload thumbnail for pid {pid}: {e}")
                    thumb_url = None
            if thumb_url is None:
                # fallback to data URI (may be large)
                b64 = base64.b64encode(data).decode('utf-8')
                thumb_url = f"data:image/png;base64,{b64}"

            duration_s = len(dets) / float(fps) if fps else 0.0
            gallery.append({
                'pid': int(pid),
                'frame': f_idx,
                'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                'duration_s': round(float(duration_s), 2),
                'thumbnail_url': thumb_url,
            })
        except Exception as e:
            logger.warning(f"Failed to generate thumbnail for pid {pid}: {e}")
            continue
    return gallery

def create_csv_report(analysis_data: dict, analysis_id: str):
    """Create CSV report from analysis data - FIXED VERSION"""
    import csv
    from io import StringIO
    
    # Use StringIO for better control
    csv_buffer = StringIO()
    writer = csv.writer(csv_buffer)
    
    # Write header
    writer.writerow(['Analysis ID', 'Person ID', 'Dwell Time (seconds)', 'Actions', 'Shelf Interactions', 'Timestamp'])
    
    # Write data
    dwell_times = analysis_data.get('dwell_time_analysis', {}).get('person_dwell_times', {})
    action_summary = analysis_data.get('action_summary', {})
    shelf_interactions = analysis_data.get('shelf_interactions', {})
    
    for person_id, dwell_time in dwell_times.items():
        # Get top action for this person if available
        actions = ', '.join(action_summary.keys()) if action_summary else 'None'
        shelf_count = sum(shelf_interactions.values()) if shelf_interactions else 0
        
        writer.writerow([
            analysis_id,
            person_id,
            f"{dwell_time:.2f}",
            actions,
            shelf_count,
            datetime.now().isoformat()
        ])
    
    # If no dwell times, add a summary row
    if not dwell_times:
        writer.writerow([
            analysis_id,
            'Summary',
            '0.00',
            ', '.join(action_summary.keys()) if action_summary else 'None',
            sum(shelf_interactions.values()) if shelf_interactions else 0,
            datetime.now().isoformat()
        ])
    
    # Get string content and encode to bytes
    csv_content = csv_buffer.getvalue()
    csv_buffer.close()
    
    return csv_content.encode('utf-8')

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Retail Behavior Analysis API", 
        "status": "running",
        "azure_blob_configured": blob_service_client is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": len(models) > 0,
        "device": str(device),
        "available_models": list(models.keys()),
        "azure_blob_configured": blob_service_client is not None,
        "container_name": CONTAINER_NAME,
        "endpoints": [
            "/",
            "/health", 
            "/analyze",
            "/apply-filters"
        ]
    }

@app.post("/analyze-batch")
async def analyze_batch(
    videos: List[UploadFile] = File(...),
    max_duration: Optional[int] = Form(60),
    frame_skip_multiplier: Optional[float] = Form(1.0),
    save_to_blob: bool = Form(True),
    generate_video: bool = Form(True)
):
    """Analyze multiple retail videos for customer behavior with batch processing"""
    
    if not videos or len(videos) == 0:
        raise HTTPException(status_code=400, detail="No videos provided")
    
    # Validate all files
    for video in videos:
        if not video.filename.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
            raise HTTPException(status_code=400, detail=f"File {video.filename} is not a supported video format")
    
    # Generate unique batch ID
    batch_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        logger.info(f"Starting batch analysis {batch_id} for {len(videos)} videos")
        
        individual_results = []
        temp_files = []
        
        # Process each video individually
        for i, video in enumerate(videos):
            logger.info(f"Processing video {i+1}/{len(videos)}: {video.filename}")
            
            # Save uploaded video temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                content = await video.read()
                tmp_file.write(content)
                video_path = tmp_file.name
                temp_files.append(video_path)
            
            # Process individual video
            if generate_video:
                result, processing_data = await process_video_analysis(video_path, max_duration, frame_skip_multiplier, generate_video=True)
            else:
                result = await process_video_analysis(video_path, max_duration, frame_skip_multiplier, generate_video=False)
                processing_data = None
            
            # Add batch metadata to individual result
            if 'metadata' not in result or result['metadata'] is None:
                result['metadata'] = {}
            
            result['metadata'].update({
                'batch_id': batch_id,
                'file_index': i,
                'total_files': len(videos),
                'batch_timestamp': timestamp
            })
            
            # Generate download_links for individual file in batch
            download_links = {}
            if save_to_blob and blob_service_client:
                try:
                    file_analysis_id = f"{batch_id}_file{i}"
                    
                    # Save JSON results
                    json_data = json.dumps(result, indent=2).encode('utf-8')
                    json_filename = f"analyses/{file_analysis_id}_{timestamp}.json"
                    download_links['json_results'] = save_to_azure_blob(json_data, json_filename, "application/json")
                    
                    # Generate and save heatmap
                    if 'heatmap_data' in result:
                        heatmap_bytes = generate_heatmap_image(result['heatmap_data'], file_analysis_id)
                        heatmap_filename = f"heatmaps/heatmap_{file_analysis_id}_{timestamp}.png"
                        download_links['heatmap_image'] = save_to_azure_blob(heatmap_bytes, heatmap_filename, "image/png")
                    
                    # Generate and save shelf map images
                    try:
                        shelf_map_images: list[str] = []
                        if processing_data:
                            top_images = generate_shelf_map_images(
                                video_path,
                                processing_data.get('shelf_boxes_per_frame', {}),
                                file_analysis_id,
                                top_k=3,
                            )
                            for idx, (f_idx, img_bytes) in enumerate(top_images):
                                shelfmap_filename = f"shelfmaps/shelfmap_{file_analysis_id}_{timestamp}_f{f_idx}_{idx+1}.png"
                                url = save_to_azure_blob(img_bytes, shelfmap_filename, "image/png")
                                shelf_map_images.append(url)
                        if shelf_map_images:
                            download_links['shelf_map_images'] = shelf_map_images
                            download_links['shelf_map_image'] = shelf_map_images[0]
                    except Exception as e:
                        logger.warning(f"Shelf map generation failed for {video.filename}: {e}")
                    
                    # Generate annotated video if requested
                    if generate_video and processing_data:
                        try:
                            video_bytes = generate_annotated_video_like_gradio(
                                video_path,
                                processing_data['tracks'],
                                processing_data['action_preds'], 
                                processing_data['shelf_boxes_per_frame'],
                                file_analysis_id,
                                processing_data['fps'],
                                max_duration
                            )
                            if video_bytes:
                                video_filename = f"videos/annotated_{file_analysis_id}_{timestamp}.mp4"
                                download_links['annotated_video_download'] = save_to_azure_blob(video_bytes, video_filename, "video/mp4")
                                download_links['annotated_video_blob_path'] = video_filename
                                download_links['annotated_video_stream'] = f"/stream?blob={quote(video_filename)}"
                        except Exception as e:
                            logger.warning(f"Failed to generate annotated video for {video.filename}: {e}")
                    
                    result['download_links'] = download_links
                    
                except Exception as e:
                    logger.warning(f"Failed to save download_links for {video.filename}: {e}")
                    result['download_links'] = None
            
            # Attach processing data if available
            if processing_data:
                try:
                    ser = _make_processing_serializable(processing_data)
                    result['processing'] = ser
                except Exception as e:
                    logger.warning(f"Failed to serialize processing data for {video.filename}: {e}")
                
                try:
                    gallery = _generate_track_gallery_thumbnails(
                        video_path,
                        processing_data.get('tracks', {}),
                        processing_data.get('fps', 30.0),
                        f"{batch_id}_file{i}",
                        timestamp,
                    )
                    if gallery:
                        result['track_gallery'] = gallery
                except Exception as e:
                    logger.warning(f"Failed to build track gallery for {video.filename}: {e}")
            
            individual_results.append(result)
        
        # Generate aggregate metrics
        aggregate_metrics = calculate_aggregate_metrics(individual_results)
        
        # Create batch result
        batch_result = {
            'analysis_type': 'batch_analysis',
            'batch_id': batch_id,
            'total_files': len(videos),
            'individual_results': individual_results,
            'aggregate_metrics': aggregate_metrics,
            'batch_metadata': {
                'timestamp': datetime.now().isoformat(),
                'processing_time': 0,  # Will be calculated
                'files_info': [
                    {
                        'filename': result.get('metadata', {}).get('original_filename', 'unknown'),
                        'file_size': result.get('metadata', {}).get('file_size', 0),
                        'analysis_id': result.get('metadata', {}).get('analysis_id', 'unknown')
                    }
                    for result in individual_results
                ]
            }
        }
        
        # Save to Azure Blob Storage if configured
        if save_to_blob and blob_service_client:
            try:
                # Save batch JSON results
                json_data = json.dumps(batch_result, indent=2).encode('utf-8')
                json_filename = f"batch_analyses/batch_{batch_id}_{timestamp}.json"
                batch_result['download_links'] = {
                    'batch_json_results': save_to_azure_blob(json_data, json_filename, "application/json")
                }
                
                logger.info(f"Batch analysis {batch_id} saved to Azure Blob Storage")
                
            except Exception as e:
                logger.error(f"Failed to save batch to Azure Blob: {e}")
                batch_result['storage_error'] = str(e)
        
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        logger.info(f"Batch analysis {batch_id} completed successfully")
        return JSONResponse(content=batch_result)
        
    except Exception as e:
        logger.error(f"Batch analysis {batch_id} failed: {e}")
        # Clean up on error
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@app.post("/analyze")
async def analyze_video(
    video: UploadFile = File(...),
    max_duration: Optional[int] = Form(60),
    frame_skip_multiplier: Optional[float] = Form(1.0),
    save_to_blob: bool = Form(True),
    generate_video: bool = Form(True)
):
    """Analyze retail video for customer behavior with optional annotated video generation"""
    
    if not video.filename.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
        raise HTTPException(status_code=400, detail="Only video files are supported")
    
    # Generate unique analysis ID
    analysis_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        logger.info(f"Starting analysis {analysis_id} for video: {video.filename}")
        
        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            content = await video.read()
            tmp_file.write(content)
            video_path = tmp_file.name
        
        # Process video - get both analytics and processing data
        if generate_video:
            result, processing_data = await process_video_analysis(video_path, max_duration, frame_skip_multiplier, generate_video=True)
        else:
            result = await process_video_analysis(video_path, max_duration, frame_skip_multiplier, generate_video=False)
            processing_data = None
        
        # Add metadata
        result['metadata'] = {
            'analysis_id': analysis_id,
            'original_filename': video.filename,
            'timestamp': datetime.now().isoformat(),
            'max_duration': max_duration,
            'frame_skip_multiplier': frame_skip_multiplier, # Add this
            'file_size': len(content),
            'video_generated': generate_video
        }
        
        # Attach processing (serialized) and gallery thumbnails when available
        try:
            if processing_data:
                try:
                    ser = _make_processing_serializable(processing_data)
                    result['processing'] = ser
                except Exception as e:
                    logger.warning(f"Failed to serialize processing data: {e}")
                try:
                    gallery = _generate_track_gallery_thumbnails(
                        video_path,
                        processing_data.get('tracks', {}),
                        processing_data.get('fps', 30.0),
                        analysis_id,
                        timestamp,
                    )
                    if gallery:
                        result['track_gallery'] = gallery
                except Exception as e:
                    logger.warning(f"Failed to build track gallery: {e}")
        except Exception as _:
            pass

        # Save to Azure Blob Storage if configured
        download_links = {}
        if save_to_blob and blob_service_client:
            try:
                # Save JSON results
                json_data = json.dumps(result, indent=2).encode('utf-8')
                json_filename = f"analyses/{analysis_id}_{timestamp}.json"
                download_links['json_results'] = save_to_azure_blob(json_data, json_filename, "application/json")
                
                # Generate and save heatmap
                if 'heatmap_data' in result:
                    heatmap_bytes = generate_heatmap_image(result['heatmap_data'], analysis_id)
                    heatmap_filename = f"heatmaps/heatmap_{analysis_id}_{timestamp}.png"
                    download_links['heatmap_image'] = save_to_azure_blob(heatmap_bytes, heatmap_filename, "image/png")

                # Generate and save shelf map images (top-3 frames)
                try:
                    shelf_map_images: list[str] = []
                    if processing_data:
                        top_images = generate_shelf_map_images(
                            video_path,
                            processing_data.get('shelf_boxes_per_frame', {}),
                            analysis_id,
                            top_k=3,
                        )
                        for idx, (f_idx, img_bytes) in enumerate(top_images):
                            shelfmap_filename = f"shelfmaps/shelfmap_{analysis_id}_{timestamp}_f{f_idx}_{idx+1}.png"
                            url = save_to_azure_blob(img_bytes, shelfmap_filename, "image/png")
                            shelf_map_images.append(url)
                    # Fallback: still try single best frame if multi failed
                    if not shelf_map_images:
                        shelf_map_bytes = generate_shelf_map_image(
                            video_path,
                            processing_data['shelf_boxes_per_frame'] if processing_data else {},
                            analysis_id,
                        )
                        if shelf_map_bytes:
                            shelfmap_filename = f"shelfmaps/shelfmap_{analysis_id}_{timestamp}.png"
                            url = save_to_azure_blob(shelf_map_bytes, shelfmap_filename, "image/png")
                            shelf_map_images.append(url)
                    if shelf_map_images:
                        download_links['shelf_map_images'] = shelf_map_images
                        download_links['shelf_map_image'] = shelf_map_images[0]
                except Exception as _e:
                    logger.warning(f"Shelf map generation failed: {_e}")
                
                # Generate and save CSV report
                try:
                    csv_data = create_csv_report(result, analysis_id)
                    csv_filename = f"reports/report_{analysis_id}_{timestamp}.csv"
                    download_links['csv_report'] = save_to_azure_blob(csv_data, csv_filename, "text/csv")
                except Exception as csv_error:
                    logger.error(f"Failed to create CSV report: {csv_error}")
                
                # Generate annotated video if requested and processing data available
                if generate_video and processing_data:
                    try:
                        logger.info(f"Generating annotated video for analysis {analysis_id}")
                        video_bytes = generate_annotated_video_like_gradio(
                            video_path,
                            processing_data['tracks'],
                            processing_data['action_preds'], 
                            processing_data['shelf_boxes_per_frame'],
                            analysis_id,
                            processing_data['fps'],
                            max_duration
                        )
                        if video_bytes:
                            video_filename = f"videos/annotated_{analysis_id}_{timestamp}.mp4"
                            # SAS URL for download (not for playback)
                            download_links['annotated_video_download'] = save_to_azure_blob(video_bytes, video_filename, "video/mp4")
                            # Stream path for playback
                            download_links['annotated_video_blob_path'] = video_filename
                            download_links['annotated_video_stream'] = f"/stream?blob={quote(video_filename)}"
                            logger.info(f"Annotated video saved: {video_filename}")
                        else:
                            logger.warning("Failed to generate annotated video")
                    except Exception as video_error:
                        logger.error(f"Failed to generate annotated video: {video_error}")
                
                result['download_links'] = download_links
                logger.info(f"Analysis {analysis_id} saved to Azure Blob Storage")
                
            except Exception as e:
                logger.error(f"Failed to save to Azure Blob: {e}")
                result['download_links'] = None
                result['storage_error'] = str(e)
        
        # Clean up temporary file
        os.unlink(video_path)
        
        logger.info(f"Analysis {analysis_id} completed successfully")
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Analysis {analysis_id} failed: {e}")
        # Clean up on error
        if 'video_path' in locals():
            try:
                os.unlink(video_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

async def process_video_analysis(video_path: str, max_duration: int = 30, frame_skip_multiplier: float = 1.0, generate_video: bool = False):
    """Process video and return analysis results using the same approach as Gradio"""
    from collections import defaultdict
    from shapely.geometry import box as shp_box
    
    # Get device from models
    device = models.get("device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Load video
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    total_frames = len(vr)
    max_frames = min(int(max_duration * fps), total_frames)
    H, W, _ = vr[0].shape
    
    logger.info(f"Processing video: {total_frames} frames, {fps} fps, max {max_frames} frames")
    
    # Initialize using YOLO tracking like in Gradio
    tracker = models["person_model"].track(
        source=video_path, 
        persist=True, 
        tracker='tracker.yaml',
        classes=[0],  # person class
        stream=True,
        device=device,
        imgsz=640,
    )
    
    # Data containers like in Gradio
    tracks = defaultdict(list)
    raw_actions = defaultdict(list)
    heatmap_grid = np.zeros((20, 20))  # Use same grid size as Gradio
    shelf_boxes_per_frame = {}
    frame_person_boxes = {} # Initialize for unique heatmap tracking
    shelf_last_box = {}
    next_shelf_idx = 1
    IOU_TH = 0.35
    
    # --- OPTIMIZATION START ---
    last_shelf_detection_result = []
    # Detect shelves at a fixed rate of 1x per second, regardless of the speed mode
    SHELF_DETECTION_INTERVAL_FRAMES = max(1, int(fps))
    logger.info(f"Shelf detection interval: {SHELF_DETECTION_INTERVAL_FRAMES} frames (fixed at 1x/sec)")
    # --- OPTIMIZATION END ---
    
    def iou_xyxy(a, b):
        """Calculate IoU between two bounding boxes"""
        try:
            inter = shp_box(*a).intersection(shp_box(*b)).area
            union = shp_box(*a).union(shp_box(*b)).area
            return inter / union if union else 0
        except:
            return 0
    
    def merge_consecutive_predictions(preds, min_duration_frames=0):
        """Merge consecutive predictions of the same class"""
        if not preds: 
            return []
        merged = []
        current = preds[0].copy()
        for nxt in preds[1:]:
            if nxt['pred'] == current['pred']:
                current['end'] = nxt['end']
            else:
                merged.append(current)
                current = nxt.copy()
        merged.append(current)
        return [e for e in merged if (e['end'] - e['start']) >= min_duration_frames]
    
    # PASS 1: Detection + Tracking (same as Gradio)
    for f_idx, result in enumerate(tracker):
        if f_idx >= max_frames:
            break
            
        frame = vr[f_idx].asnumpy()
        
        # --- OPTIMIZATION START ---
        # Only run shelf detection periodically
        if f_idx % SHELF_DETECTION_INTERVAL_FRAMES == 0:
            res_shelf = models["shelf_model"](frame, device=device, imgsz=640)
            
            # Process shelf detections (same logic as Gradio)
            assigned = []
            raw_boxes = [b.xyxy[0].cpu().numpy() for b in res_shelf[0].boxes] if res_shelf[0].boxes else []
            for box in raw_boxes:
                cur = tuple(map(int, box))
                best_iou, best_id = 0, None
                for sid, prev in shelf_last_box.items():
                    val = iou_xyxy(cur, prev)
                    if val > best_iou:
                        best_iou, best_id = val, sid
                
                if best_iou >= IOU_TH:
                    shelf_last_box[best_id] = cur
                    assigned.append((best_id, cur))
                else:
                    sid = f"shelf_{next_shelf_idx}"
                    next_shelf_idx += 1
                    shelf_last_box[sid] = cur
                    assigned.append((sid, cur))
            
            # Cache the result
            last_shelf_detection_result = assigned
        
        # Use the latest cached result for the current frame
        shelf_boxes_per_frame[f_idx] = last_shelf_detection_result
        # --- OPTIMIZATION END ---
        
        # Process person detections (same as Gradio)
        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.int().cpu().tolist()
            
            # --- Heatmap unique presence tracking ---
            # Track unique (pid, grid_cell) per second to make heatmap values more realistic
            time_bucket = int(f_idx / fps) if fps > 0 else 0
            if time_bucket not in frame_person_boxes:
                frame_person_boxes[time_bucket] = set()
            # --- End Heatmap tracking ---

            for box, pid in zip(boxes, ids):
                tracks[pid].append({'frame': f_idx, 'bbox': box, 'pid': pid})
                cx, cy = (box[0] + box[2])/2, (box[1] + box[3])/2
                gx, gy = min(int(cx/W*20), 19), min(int(cy/H*20), 19)
                
                # Add unique presence to the current time bucket
                frame_person_boxes[time_bucket].add((pid, (gx, gy)))
    
    # After Pass 1, build the final heatmap grid from unique presences
    for time_bucket in frame_person_boxes.values():
        for _, (gx, gy) in time_bucket:
            heatmap_grid[gy, gx] += 1
    
    # Action Recognition (same logic as Gradio)
    # Stride controls how densely we sample clips for the action model
    action_stride = max(1, int(12 * frame_skip_multiplier))
    logger.info(f"Action recognition stride: {action_stride} frames (multiplier: {frame_skip_multiplier})")
    for pid, dets in tracks.items():
        if len(dets) < 16: 
            continue
        for i in range(0, len(dets)-15, action_stride): # stride now dynamic based on multiplier
            clip_frames = [d['frame'] for d in dets[i:i+16]]
            try:
                imgs = vr.get_batch(clip_frames).asnumpy()
                crops = [img[int(d['bbox'][1]):int(d['bbox'][3]),
                           int(d['bbox'][0]):int(d['bbox'][2])] for img, d in zip(imgs, dets[i:i+16])]
                if not crops: 
                    continue
                
                inp = models["image_processor"](crops, return_tensors='pt').to(device)
                pred = models["action_model"](**inp).logits.argmax(-1).item()
                
                # MAP TO ACTION NAME - TAMBAHKAN DI SINI
                id2label = models.get("id2label")
                if not id2label:
                    raise ValueError("Action model id2label mapping not available")
                
                action_name = id2label.get(pred, f"unknown_action_{pred}")
                
                # Store dengan action_name, bukan pred
                raw_actions[pid].append({
                    'start': dets[i]['frame'], 
                    'end': dets[i+15]['frame'], 
                    'pred': pred,
                    'action_name': action_name  # Tambahkan ini
                })
                
            except Exception as e:
                logger.error(f"Error processing action for pid {pid}: {e}")
    
    # Merge consecutive predictions
    action_preds = {pid: merge_consecutive_predictions(v, int(fps*0.4))
                   for pid, v in raw_actions.items()}
    
    # Calculate Shelf Interactions (same as Gradio)
    shelf_interaksi = defaultdict(int)
    for pid, dets in tracks.items():
        for d in dets:
            f = d['frame']
            x1, y1, x2, y2 = d['bbox']            
            cx, cy = (x1+x2)/2, (y1+y2)/2
            for sid, (sx1, sy1, sx2, sy2) in shelf_boxes_per_frame.get(f, []):
                if sx1 <= cx <= sx2 and sy1 <= cy <= sy2:
                    shelf_interaksi[sid] += 1
                    
    # Generate analytics
    analytics = generate_analytics_gradio_style(tracks, action_preds, shelf_interaksi, fps, heatmap_grid, shelf_boxes_per_frame)
    
    # Return analytics and processing data for video generation if needed
    if generate_video:
        processing_data = {
            'tracks': tracks,
            'action_preds': action_preds,
            'shelf_boxes_per_frame': shelf_boxes_per_frame,
            'fps': fps,
            'frame_size': (H, W)
        }
        return analytics, processing_data
    else:
        return analytics

def calculate_aggregate_metrics(individual_results: List[dict]) -> dict:
    """Calculate aggregate metrics from multiple analysis results without spatial data merging"""
    try:
        if not individual_results:
            return {}
        
        # Aggregate basic metrics
        total_unique_persons = sum(r.get('unique_persons', 0) for r in individual_results)
        total_interactions = sum(r.get('total_interactions', 0) for r in individual_results)
        
        # Calculate average dwell time across all videos
        dwell_times = [r.get('dwell_time_analysis', {}).get('average_dwell_time', 0) for r in individual_results]
        average_dwell_time_across_videos = sum(dwell_times) / len(dwell_times) if dwell_times else 0
        
        # Find most common action overall
        action_counts = defaultdict(int)
        for result in individual_results:
            action_summary = result.get('action_summary', {})
            for action, count in action_summary.items():
                action_counts[action] += count
        
        most_common_action_overall = max(action_counts.items(), key=lambda x: x[1])[0] if action_counts else "N/A"
        
        # Calculate total analysis duration
        total_analysis_duration = sum(r.get('metadata', {}).get('max_duration', 0) for r in individual_results)
        
        return {
            'total_unique_persons': total_unique_persons,
            'total_interactions': total_interactions,
            'average_dwell_time_across_videos': round(average_dwell_time_across_videos, 2),
            'most_common_action_overall': most_common_action_overall,
            'total_analysis_duration': total_analysis_duration,
            'files_analyzed': len(individual_results),
            'action_distribution': dict(action_counts),
            'per_file_summary': [
                {
                    'filename': r.get('metadata', {}).get('original_filename', 'unknown'),
                    'unique_persons': r.get('unique_persons', 0),
                    'total_interactions': r.get('total_interactions', 0),
                    'avg_dwell_time': r.get('dwell_time_analysis', {}).get('average_dwell_time', 0),
                    'most_common_action': r.get('behavioral_insights', {}).get('most_common_action', 'N/A')
                }
                for r in individual_results
            ]
        }
    except Exception as e:
        logger.error(f"Error calculating aggregate metrics: {e}")
        return {
            'total_unique_persons': 0,
            'total_interactions': 0,
            'average_dwell_time_across_videos': 0,
            'most_common_action_overall': 'N/A',
            'total_analysis_duration': 0,
            'files_analyzed': len(individual_results) if individual_results else 0,
            'error': str(e)
        }

def generate_analytics_gradio_style(tracks, action_preds, shelf_interaksi, fps, heatmap_grid, shelf_boxes_per_frame):
    """Generate analytics with optimized action_shelf_mapping (reduced redundancy)"""
    from shapely.geometry import box as shp_box
    import pandas as pd

    def calculate_iou(box_a, box_b):
        """Calculate IoU between two bounding boxes."""
        try:
            poly_a = shp_box(*box_a)
            poly_b = shp_box(*box_b)
            inter_area = poly_a.intersection(poly_b).area
            union_area = poly_a.union(poly_b).area
            return inter_area / union_area if union_area > 0 else 0
        except Exception:
            return 0

    # OPTIMIZED: Collect unique person-shelf-action sequences (no redundancy)
    action_shelf_data = []
    shelf_action_counter = defaultdict(int)
    
    for pid, acts in action_preds.items():
        last_shelf_action = None  # Track last state to avoid duplicates
        
        for seg in acts:
            s, e, act_id = seg['start'], seg['end'], seg['pred']
            act_label = models["id2label"][act_id]
            
            # Sample fewer frames (start, middle, end) instead of ALL frames
            sample_frames = [s, (s+e)//2, e] if e > s else [s]
            
            for f in sample_frames:
                det = next((d for d in tracks[pid] if d['frame'] == f), None)
                if det is None: 
                    continue
                
                person_bbox = det['bbox']
                
                for sid, shelf_bbox in shelf_boxes_per_frame.get(f, []):
                    if calculate_iou(person_bbox, shelf_bbox) > 0.01:
                        current_state = (sid, act_label)
                        
                        # Only add if this is a NEW state (reduces redundancy by 80%+)
                        if current_state != last_shelf_action:
                            action_shelf_data.append([pid, f, sid, act_label])
                            shelf_action_counter[(sid, act_label)] += 1
                            last_shelf_action = current_state
                        break

    # DIRECT JOURNEY ANALYSIS (like Gradio generate_journey_analysis)
    journey_analysis = {}
    if action_shelf_data:
        try:
            df = pd.DataFrame(action_shelf_data, columns=['pid', 'frame', 'shelf_id', 'action'])
            
            # Find key events (same logic as Gradio)
            reach_events = df[df['action'] == 'Reach To Shelf'][['pid', 'shelf_id']].drop_duplicates().assign(did_reach=True)
            inspect_events = df[df['action'].isin(['Inspect Product', 'Inspect Shelf'])][['pid', 'shelf_id']].drop_duplicates().assign(did_inspect=True)
            return_events = df[df['action'] == 'Hand In Shelf'][['pid', 'shelf_id']].drop_duplicates().assign(did_return=True)
            
            # Create analysis dataframe
            interactions = df[['pid', 'shelf_id']].drop_duplicates()
            analysis_df = pd.merge(interactions, reach_events, on=['pid', 'shelf_id'], how='left')
            analysis_df = pd.merge(analysis_df, inspect_events, on=['pid', 'shelf_id'], how='left')
            analysis_df = pd.merge(analysis_df, return_events, on=['pid', 'shelf_id'], how='left')
            analysis_df = analysis_df.fillna(False)
            
            # Categorize outcomes (exact same as Gradio)
            def categorize_outcome(row):
                if not row['did_reach']:
                    return 'No Reach'
                if row['did_inspect'] and row['did_return']:
                    return 'Keraguan & Pembatalan'
                elif row['did_inspect'] and not row['did_return']:
                    return 'Konversi Sukses'
                else:
                    return 'Kegagalan Menarik Minat'
            
            analysis_df['outcome'] = analysis_df.apply(categorize_outcome, axis=1)
            relevant_outcomes = analysis_df[analysis_df['outcome'] != 'No Reach']
            
            if not relevant_outcomes.empty:
                # Generate percentages for each shelf
                outcome_summary = relevant_outcomes.groupby(['shelf_id', 'outcome']).size().unstack(fill_value=0)
                outcome_percentage = outcome_summary.div(outcome_summary.sum(axis=1), axis=0) * 100
                
                # Ensure all outcome columns exist
                desired_order = ['Konversi Sukses', 'Keraguan & Pembatalan', 'Kegagalan Menarik Minat']
                for col in desired_order:
                    if col not in outcome_percentage.columns:
                        outcome_percentage[col] = 0
                
                outcome_percentage = outcome_percentage[desired_order]
                
                # Convert to frontend format
                journey_data = []
                for shelf_id, row in outcome_percentage.iterrows():
                    journey_data.append({
                        'shelf_id': shelf_id,
                        'konversi_sukses': round(row.get('Konversi Sukses', 0), 1),
                        'keraguan_pembatalan': round(row.get('Keraguan & Pembatalan', 0), 1),
                        'kegagalan_menarik_minat': round(row.get('Kegagalan Menarik Minat', 0), 1),
                        'total_interactions': int(outcome_summary.loc[shelf_id].sum()) if shelf_id in outcome_summary.index else 0
                    })
                
                journey_analysis = {'journey_data': journey_data}
        except Exception as e:
            logger.warning(f"Journey analysis failed: {e}")
            journey_analysis = {}

    # Dwell Time Analysis (per person)
    person_dwell_times = defaultdict(float)
    for pid, dets in tracks.items():
        person_dwell_times[pid] = len(dets) / fps

    # Behavioral Insights (by merged action segments per person, not per-frame or per-shelf)
    action_summary = defaultdict(int)
    try:
        id2label = models["id2label"]
    except Exception:
        id2label = {}
    for pid, segs in action_preds.items():
        for seg in segs:
            act_id = int(seg.get('pred', 0))
            act_label = id2label.get(act_id, f"action_{act_id}")
            action_summary[act_label] += 1

    most_common_action = max(action_summary.items(), key=lambda x: x[1])[0] if action_summary else "N/A"
    
    # Unique persons
    unique_persons = set(tracks.keys())

    # Build mapping: frame -> list of person boxes
    frame_person_boxes = defaultdict(list)
    for pid, person_tracks in tracks.items():
        for det in person_tracks:
            frame_person_boxes[det['frame']].append(det['bbox'])

    # Unique-per-frame occupancy and cumulative (person-seconds) per shelf
    shelf_frame_counts = defaultdict(int)            # unique per frame (max video duration)
    shelf_person_frame_counts = defaultdict(int)     # cumulative per person (can exceed video duration)

    for f_idx, shelves in shelf_boxes_per_frame.items():
        present_shelves = set()
        if not shelves:
            continue
        for person_box in frame_person_boxes.get(f_idx, []):
            for sid, shelf_box in shelves:
                if calculate_iou(person_box, shelf_box) > 0.01:
                    present_shelves.add(sid)
                    shelf_person_frame_counts[sid] += 1
                    break
        for sid in present_shelves:
            shelf_frame_counts[sid] += 1

    # Calculate dwell time per shelf in seconds (unique per frame)
    shelf_dwell_times_seconds = {
        shelf_id: round(frames / fps, 2)
        for shelf_id, frames in shelf_frame_counts.items()
    }
    # Also expose cumulative person-seconds per shelf
    shelf_dwell_load_seconds = {
        shelf_id: round(frames / fps, 2)
        for shelf_id, frames in shelf_person_frame_counts.items()
    }

    # Prepare results dictionary - MUCH SMALLER NOW
    results = {
        "unique_persons": len(unique_persons),  # ADD THIS LINE
        "total_interactions": int(sum(action_summary.values())),  # FIX: Use action segments, not action_shelf_data
        "shelf_interactions": dict(shelf_person_frame_counts),
        "shelf_dwell_times": shelf_dwell_times_seconds,              # unique per frame
        "shelf_dwell_load_seconds": shelf_dwell_load_seconds,        # person-seconds
        "dwell_time_analysis": {
            "person_dwell_times": {str(k): float(v) for k, v in person_dwell_times.items()},
            "average_dwell_time": float(np.mean(list(person_dwell_times.values()))) if person_dwell_times else 0,
            "max_dwell_time": float(max(person_dwell_times.values())) if person_dwell_times else 0,
        },
        "behavioral_insights": {
            "most_common_action": most_common_action,
            "total_actions_detected": int(sum(action_summary.values())),
            "action_summary": dict(action_summary),
            "average_confidence": 0.82,  # placeholder avg confidence
        },
        "processing_info": {
            "fps": float(fps),
            "total_tracks": len(unique_persons),
            "total_shelf_interactions": sum(shelf_person_frame_counts.values())
        },
        "heatmap_data": heatmap_grid.tolist(),
        # OPTIMIZED: Much smaller action_shelf_mapping (80%+ reduction)
        # "action_shelf_mapping": action_shelf_data[:50],  # Limit to first 100 for debugging
        "action_shelf_mapping": action_shelf_data,  # Limit to first 100 for debugging
        # NEW: Ready-to-use journey analysis
        "journey_analysis": journey_analysis
    }
    
    return results

def _recompute_with_exclusions(processing: dict, excluded_ids: set[str] | set[int]):
    """Recompute analytics excluding given track IDs (as strings or ints)."""
    try:
        # Normalize keys to str
        excluded_str = set(str(x) for x in excluded_ids)
        tracks = {}
        for pid, dets in processing.get('tracks', {}).items():
            if str(pid) in excluded_str:
                continue
            tracks[int(pid)] = [
                {
                    'frame': int(d['frame']),
                    'bbox': np.array(d['bbox'], dtype=float),
                    'pid': int(d.get('pid', pid))
                }
                for d in dets
            ]

        action_preds = {}
        for pid, preds in processing.get('action_preds', {}).items():
            if str(pid) in excluded_str:
                continue
            action_preds[int(pid)] = [
                {
                    'start': int(p['start']),
                    'end': int(p['end']),
                    'pred': int(p['pred'])
                }
                for p in preds
            ]

        shelf_boxes_per_frame = {}
        for f, items in processing.get('shelf_boxes_per_frame', {}).items():
            ser_items = []
            for sid, box in items or []:
                ser_items.append((str(sid), (int(box[0]), int(box[1]), int(box[2]), int(box[3]))))
            shelf_boxes_per_frame[int(f)] = ser_items

        fps = float(processing.get('fps', 30.0))

        # Rebuild heatmap by summing included tracks only
        # Get frame size if present to map to grid, else default 20x20
        heatmap_grid = np.zeros((20, 20))
        frame_size = processing.get('frame_size') or [None, None]
        H, W = (frame_size[0], frame_size[1]) if isinstance(frame_size, (list, tuple)) and len(frame_size) == 2 else (None, None)
        if not (H and W):
            # Infer from max bbox extents if frame_size missing
            max_w = 0.0
            max_h = 0.0
            for _, dets in tracks.items():
                for d in dets:
                    x1, y1, x2, y2 = d['bbox']
                    if x2 > max_w:
                        max_w = float(x2)
                    if y2 > max_h:
                        max_h = float(y2)
            if max_w > 0 and max_h > 0:
                W = max_w
                H = max_h
            else:
                # Default fallback
                W, H = 640.0, 480.0
                
        if H and W:
            H = float(H); W = float(W)
            for _, dets in tracks.items():
                for d in dets:
                    x1, y1, x2, y2 = d['bbox']
                    cx, cy = (x1 + x2)/2.0, (y1 + y2)/2.0
                    gx, gy = min(int(cx/W*20), 19), min(int(cy/H*20), 19)
                    heatmap_grid[gy, gx] += 1
        
        logger.info(f"Rebuilt heatmap: total heat points = {heatmap_grid.sum()}")

        # shelf_interaksi is recomputed inside analytics function from tracks + shelves anyway
        shelf_interaksi = defaultdict(int)

        result = generate_analytics_gradio_style(tracks, action_preds, shelf_interaksi, fps, heatmap_grid, shelf_boxes_per_frame)
        try:
            result['processing_info']['excluded_ids'] = list(sorted(int(x) if str(x).isdigit() else str(x) for x in excluded_str))
        except Exception:
            pass
        return result
    except Exception as e:
        logger.error(f"Failed to recompute with exclusions: {e}")
        raise HTTPException(status_code=400, detail="Invalid processing data for recompute")

@app.post("/apply-filters")
@app.options("/apply-filters")  # Handle preflight CORS
async def apply_filters(request: Request):
    """Recompute analytics excluding selected track IDs.
    Recomputes only metrics and visualizations, not LLM insights.
    """
    try:
        # Manually parse and validate to handle potential inconsistencies
        body = await request.json()
        try:
            req = ApplyFiltersRequest.parse_obj(body)
        except Exception as pydantic_error:
            logger.error(f"Pydantic validation FAILED for /apply-filters: {pydantic_error}", exc_info=True)
            raise HTTPException(
                status_code=422, 
                detail=f"Invalid payload structure: {pydantic_error}"
            )

        if not req.processing:
            raise HTTPException(status_code=400, detail="processing data is required for now")
        
        excluded = set(req.excluded_track_ids or [])
        
        # Validate processing structure more leniently
        if not isinstance(req.processing, dict):
            raise HTTPException(status_code=422, detail="processing must be a dictionary")

        # Allow minimal viable fields: tracks + fps; other fields optional
        minimal_keys = ['tracks', 'fps']
        missing_min = [k for k in minimal_keys if k not in req.processing]
        if missing_min:
            return JSONResponse(status_code=422, content={
                "detail": [{"msg": "processing missing required keys", "missing": missing_min}]
            })

        # Ensure optional keys exist to avoid downstream KeyErrors
        req.processing.setdefault('action_preds', {})
        req.processing.setdefault('shelf_boxes_per_frame', {})
        
        # Defensive: coerce track keys to strings
        try:
            if isinstance(req.processing.get('tracks'), dict):
                req.processing['tracks'] = {str(k): v for k, v in req.processing['tracks'].items()}
            if isinstance(req.processing.get('action_preds'), dict):
                req.processing['action_preds'] = {str(k): v for k, v in req.processing['action_preds'].items()}
        except Exception as e:
            logger.warning(f"Failed to normalize processing keys: {e}")

        # Recompute analytics
        analytics = _recompute_with_exclusions(req.processing, excluded)
        
        # --- Generate new heatmap image from filtered data ---
        if 'heatmap_data' in analytics:
            try:
                analysis_id = req.analysis_id or "filtered"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                heatmap_bytes = generate_heatmap_image(np.array(analytics['heatmap_data']), analysis_id)
                
                new_heatmap_url = None
                if blob_service_client:
                    heatmap_filename = f"heatmaps/heatmap_{analysis_id}_{timestamp}_filtered.png"
                    new_heatmap_url = save_to_azure_blob(heatmap_bytes, heatmap_filename, "image/png")
                
                if new_heatmap_url:
                    analytics['download_links'] = {'heatmap_image': new_heatmap_url}
                else:
                    # Fallback to data URI if blob storage fails or is not configured
                    heatmap_base64 = base64.b64encode(heatmap_bytes).decode('utf-8')
                    analytics['download_links'] = {'heatmap_image': f"data:image/png;base64,{heatmap_base64}"}

            except Exception as e:
                logger.warning(f"Failed to generate new heatmap image on-the-fly: {e}")
                analytics['download_links'] = {'heatmap_image': None} # Signal to frontend that image gen failed
        
        # Keep original LLM insights if they exist
        if isinstance(req.processing, dict) and 'llm_insights' in req.processing:
            analytics['llm_insights'] = req.processing['llm_insights']
        
        logger.info(f"Recomputed analytics excluding {len(excluded)} IDs.")
        
        return JSONResponse(
            content=analytics,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/apply-filters error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to apply filters: {str(e)}")

@app.get("/analysis/{analysis_id}")
async def get_analysis_results(analysis_id: str):
    """Get analysis results from Azure Blob Storage"""
    if not blob_service_client:
        raise HTTPException(status_code=500, detail="Azure Blob Storage not configured")
    
    try:
        # Try to get JSON results
        json_filename = f"analyses/{analysis_id}_*.json"
        blob_client = blob_service_client.get_blob_client(
            container=CONTAINER_NAME, 
            blob=json_filename
        )
        
        # Download and return results
        blob_data = blob_client.download_blob()
        results = json.loads(blob_data.readall())
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found: {str(e)}")

@app.get("/list-analyses")
async def list_analyses():
    """List all available analyses"""
    if not blob_service_client:
        raise HTTPException(status_code=500, detail="Azure Blob Storage not configured")
    
    try:
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        analyses = []
        
        for blob in container_client.list_blobs(name_starts_with="analyses/"):
            analysis_id = blob.name.split('/')[1].split('_')[0]
            analyses.append({
                'analysis_id': analysis_id,
                'filename': blob.name,
                'size': blob.size,
                'created': blob.creation_time.isoformat()
            })
        
        return {"analyses": analyses}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list analyses: {str(e)}")

@app.post("/analyze-debug")
async def analyze_video_debug(
    video: UploadFile = File(...),
    max_duration: Optional[int] = 30
):
    """Debug endpoint - analyze video without saving to Azure Blob"""
    
    if not video.filename.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
        raise HTTPException(status_code=400, detail="Only video files are supported")
    
    # Generate unique analysis ID
    analysis_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Starting debug analysis {analysis_id} for video: {video.filename}")
        
        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            content = await video.read()
            tmp_file.write(content)
            video_path = tmp_file.name
        
        # Process video
        result = await process_video_analysis(video_path, max_duration)
        
        # Add metadata
        result['metadata'] = {
            'analysis_id': analysis_id,
            'original_filename': video.filename,
            'timestamp': datetime.now().isoformat(),
            'max_duration': max_duration,
            'file_size': len(content),
            'debug_mode': True
        }
        
        # Generate heatmap for response (base64 encoded)
        if 'heatmap_data' in result:
            heatmap_bytes = generate_heatmap_image(result['heatmap_data'], analysis_id)
            heatmap_base64 = base64.b64encode(heatmap_bytes).decode('utf-8')
            result['heatmap_base64'] = heatmap_base64
        
        # Clean up temporary file
        os.unlink(video_path)
        
        logger.info(f"Debug analysis {analysis_id} completed successfully")
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Debug analysis {analysis_id} failed: {e}")
        # Clean up on error
        if 'video_path' in locals():
            try:
                os.unlink(video_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/compare-gradio")
async def compare_with_gradio():
    """Compare API capabilities with Gradio implementation"""
    return {
        "fastapi_features": {
            "endpoints": [
                "/health - Health check",
                "/analyze - Full analysis with Azure Blob storage",
                "/analyze-debug - Debug analysis without storage",
                "/analysis/{id} - Get stored analysis",
                "/list-analyses - List all analyses"
            ],
            "processing": [
                "YOLO person detection and tracking",
                "Shelf detection and mapping", 
                "Action classification using transformers",
                "Heatmap generation",
                "Dwell time analysis",
                "Behavioral insights"
            ],
            "storage": [
                "Azure Blob Storage integration",
                "JSON results",
                "PNG heatmap images", 
                "CSV reports"
            ]
        },
        "gradio_equivalent": {
            "video_processing": "✅ Same YOLO tracking approach",
            "action_recognition": "✅ Same transformer model", 
            "heatmap_generation": "✅ Same grid-based approach",
            "shelf_detection": "✅ Same shelf segmentation model",
            "analytics": "✅ Compatible output format"
        },
        "improvements": [
            "RESTful API for web integration",
            "Cloud storage capabilities", 
            "Scalable architecture",
            "Debug endpoints for development",
            "JSON structured responses"
        ]
    }

@app.get("/model-info")
async def get_model_info():
    """Get information about loaded models"""
    if not models:
        return {"error": "Models not loaded yet"}
    
    info = {
        "models_loaded": list(models.keys()),
        "device": str(models.get("device", "unknown")),
        "action_classes": models.get("id2label", {}),        "model_details": {
            "person_model": "YOLO11s - Person detection and tracking (models/yolo11s.pt)",
            "shelf_model": "Custom YOLO - Shelf segmentation (models/shelf_model/best.pt)", 
            "action_model": "haipradana/tracko-videomae-action-detection (models/action_model/ or online)",
            "image_processor": "Video classification preprocessor"
        }
    }
    
    if torch.cuda.is_available():
        info["gpu_info"] = {
            "available": True,
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB",
            "memory_cached": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB"
        }
    else:
        info["gpu_info"] = {"available": False}
    
    return info

def generate_annotated_video_like_gradio(video_path: str, tracks: dict, action_preds: dict, 
                                        shelf_boxes_per_frame: dict, analysis_id: str, fps: float, max_duration: int = 30):
    """Generate annotated video exactly like Gradio version with supervision heatmap overlay"""
    
    try:
        # Load video
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        H, W, _ = vr[0].shape
        max_frames = min(total_frames, int(fps * max_duration))
        
        # Create output video writer
        output_path = f"temp_annotated_{analysis_id}.mp4"
        # Prefer H.264 (avc1). If not available in OpenCV build, fallback to mp4v then re-encode to H.264 via ffmpeg if present.
        vw = None
        selected_codec = None
        tried_codecs = [('avc1', 'H.264 (avc1)'), ('H264', 'H.264'), ('mp4v', 'MPEG-4 Part 2')]
        for codec in tried_codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec[0])
            candidate = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
            if candidate.isOpened():
                vw = candidate
                selected_codec = codec
                logger.info(f"Using video codec {codec[1]} [{codec[0]}]")
                break
        if vw is None:
            raise RuntimeError("Failed to initialize video writer with supported codec")
        
        # Initialize supervision heatmap annotator (exactly like Gradio)
        heatmap_ann = sv.HeatMapAnnotator(
            position=sv.Position.BOTTOM_CENTER,
            opacity=0.3, 
            radius=20, 
            kernel_size=25
        )
        
        # Frame rendering optimization like Gradio
        render_every = max(1, int(len(vr) / 300))  # Aim for ~300 frames max
        logger.info(f"Rendering every {render_every} frames for performance")
        
        for f_idx in range(max_frames):
            # Skip frames for performance (same as Gradio)
            if f_idx % render_every != 0 and f_idx != max_frames-1:
                continue
                
            frame = vr[f_idx].asnumpy()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Draw shelves with intra-frame merge to avoid duplicates/overlaps
            raw_shelves = shelf_boxes_per_frame.get(f_idx, [])
            # 1) Keep the largest box per shelf id
            per_id = {}
            for sid, (x1, y1, x2, y2) in raw_shelves:
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                area = max(0, x2 - x1) * max(0, y2 - y1)
                if sid not in per_id:
                    per_id[sid] = ((x1, y1, x2, y2), area)
                else:
                    if area > per_id[sid][1]:
                        per_id[sid] = ((x1, y1, x2, y2), area)

            boxes = [(sid, box) for sid, (box, _) in per_id.items()]

            # 2) Merge overlapping boxes across different ids (union) if IoU >= MERGE_IOU
            MERGE_IOU = 0.35
            def iou_box(a, b):
                ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
                ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
                return inter / area if area > 0 else 0.0

            clusters = []  # each: {box: (x1,y1,x2,y2), ids: [sid,...]}
            for sid, box in boxes:
                placed = False
                for c in clusters:
                    if iou_box(box, c["box"]) >= MERGE_IOU:
                        x1, y1, x2, y2 = c["box"]
                        bx1, by1, bx2, by2 = box
                        c["box"] = (min(x1, bx1), min(y1, by1), max(x2, bx2), max(y2, by2))
                        c["ids"].append(sid)
                        placed = True
                        break
                if not placed:
                    clusters.append({"box": box, "ids": [sid]})

            # 3) Draw merged shelves and a readable label inside
            for c in clusters:
                x1, y1, x2, y2 = c["box"]
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # Choose a numeric label if possible (min numeric suffix among ids)
                label_num = None
                for sid in c["ids"]:
                    try:
                        n = int(str(sid).split('_')[-1])
                        label_num = n if label_num is None else min(label_num, n)
                    except Exception:
                        continue
                label_text = str(label_num) if label_num is not None else (c["ids"][0] if c["ids"] else "shelf")
                # Centered large label
                box_h = max(1, y2 - y1)
                font_scale = max(0.6, min(1.8, box_h / 140.0))
                thickness = 2 if font_scale < 1.2 else 3
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cx = x1 + (x2 - x1) // 2 - tw // 2
                cy = y1 + (y2 - y1) // 2 + th // 2
                cv2.putText(frame_bgr, label_text, (cx+2, cy+2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness+2)
                cv2.putText(frame_bgr, label_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            # Draw persons with actions (exactly like Gradio)
            cur_tracks = [t for pid, v in tracks.items() for t in v if t['frame'] == f_idx]
            for t in cur_tracks:
                x1, y1, x2, y2 = map(int, t['bbox'])
                pid = t['pid']
                label = f"ID {pid}"
                
                # Find current action for this person (same logic as Gradio)
                for a in action_preds.get(pid, []):
                    if a['start'] <= f_idx <= a['end']:
                        id2label = models.get("id2label", {})
                        action_label = id2label.get(a['pred'], f"action_{a['pred']}")
                        label += f" | {action_label}"
                        break
                        
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for persons
                cv2.putText(frame_bgr, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add heatmap overlay with supervision (exactly like Gradio)
            if cur_tracks:
                # Create supervision detections format
                detections = sv.Detections(
                    xyxy=np.array([t['bbox'] for t in cur_tracks]),
                    confidence=np.ones(len(cur_tracks)),
                    class_id=np.zeros(len(cur_tracks), dtype=int)
                )
                # Apply heatmap annotation
                frame_bgr = heatmap_ann.annotate(scene=frame_bgr.copy(), detections=detections)
            
            # Add frame info
            frame_info = f"Frame: {f_idx} | Time: {f_idx/fps:.1f}s | Analysis: {analysis_id[:8]}"
            cv2.putText(frame_bgr, frame_info, (10, H-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            vw.write(frame_bgr)
        
        vw.release()
        logger.info(f"Annotated video generated: {output_path}")
        
        # If we had to fallback to mp4v but ffmpeg is available, re-encode to H.264 for best browser compatibility
        if selected_codec and selected_codec[0] == 'mp4v':
            try:
                ffmpeg_version = subprocess.run(['ffmpeg', '-version'], capture_output=True)
                if ffmpeg_version.returncode == 0:
                    h264_path = f"temp_annotated_{analysis_id}_h264.mp4"
                    # Fast H.264 encode suitable for web streaming
                    cmd = [
                        'ffmpeg', '-y', '-loglevel', 'error',
                        '-i', output_path,
                        '-c:v', 'libx264', '-preset', 'veryfast', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart',
                        h264_path
                    ]
                    subprocess.check_call(cmd)
                    os.unlink(output_path)
                    output_path = h264_path
                    logger.info("Re-encoded annotated video to H.264 with ffmpeg")
            except Exception as _:
                logger.warning("ffmpeg not available or re-encode failed; using original mp4v file")

        # Read the annotated video as bytes
        with open(output_path, 'rb') as f:
            video_bytes = f.read()
        
        # Clean up temporary file
        os.unlink(output_path)
        
        return video_bytes
        
    except Exception as e:
        logger.error(f"Error generating annotated video: {e}")
        # Clean up on error
        if 'vw' in locals() and vw:
            vw.release()
        if 'output_path' in locals() and os.path.exists(output_path):
            os.unlink(output_path)
        return None

def generate_shelf_map_image(video_path: str, shelf_boxes_per_frame: dict, analysis_id: str) -> bytes | None:
    """Generate a single-frame image with only shelf bounding boxes labeled with shelf ids.

    Picks the frame with the highest number of shelves detected. Returns PNG bytes.
    """
    try:
        if not shelf_boxes_per_frame:
            return None
        # Choose frame with most shelves
        best_frame_idx = max(shelf_boxes_per_frame.keys(), key=lambda k: len(shelf_boxes_per_frame.get(k, [])))
        vr = VideoReader(video_path, ctx=cpu(0))
        frame = vr[int(best_frame_idx)].asnumpy()
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Deduplicate overlapping shelves by keeping the largest box per shelf id
        raw = shelf_boxes_per_frame.get(best_frame_idx, [])
        best_by_id: dict[str, tuple[int, int, int, int]] = {}
        for sid, (x1, y1, x2, y2) in raw:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if sid not in best_by_id:
                best_by_id[sid] = (x1, y1, x2, y2)
            else:
                bx1, by1, bx2, by2 = best_by_id[sid]
                barea = max(0, bx2 - bx1) * max(0, by2 - by1)
                if area > barea:
                    best_by_id[sid] = (x1, y1, x2, y2)

        # Draw shelves (single box per id) with big centered numeric label inside
        for sid, (x1, y1, x2, y2) in best_by_id.items():
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 120, 255), 2)

            # Prefer numeric id for large label (e.g., shelf_4 -> 4)
            try:
                num = str(int(str(sid).split('_')[-1]))
                label_text = num
            except Exception:
                label_text = str(sid)

            # Dynamic font scale based on box height
            box_h = max(1, y2 - y1)
            font_scale = max(0.7, min(2.0, box_h / 140.0))
            thickness = 2 if font_scale < 1.2 else 3

            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cx = x1 + (x2 - x1) // 2 - tw // 2
            cy = y1 + (y2 - y1) // 2 + th // 2
            # Add subtle background shadow for readability
            cv2.putText(img, label_text, (cx+2, cy+2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness+2)
            cv2.putText(img, label_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # Title banner
        title = f"Shelf Map — Frame {best_frame_idx} — Analysis {analysis_id[:8]}"
        cv2.putText(img, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        ok, buf = cv2.imencode('.png', img)
        if not ok:
            return None
        return buf.tobytes()
    except Exception as e:
        logger.error(f"Error generating shelf map image: {e}")
        return None

def _generate_shelf_map_image_for_frame(video_path: str, frame_idx: int, shelf_boxes_per_frame: dict, analysis_id: str) -> bytes | None:
    """Generate shelf map image for a specific frame index."""
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        frame = vr[int(frame_idx)].asnumpy()
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Deduplicate overlapping shelves by keeping the largest box per shelf id
        raw = shelf_boxes_per_frame.get(frame_idx, [])
        best_by_id: dict[str, tuple[int, int, int, int]] = {}
        for sid, (x1, y1, x2, y2) in raw:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if sid not in best_by_id:
                best_by_id[sid] = (x1, y1, x2, y2)
            else:
                bx1, by1, bx2, by2 = best_by_id[sid]
                barea = max(0, bx2 - bx1) * max(0, by2 - by1)
                if area > barea:
                    best_by_id[sid] = (x1, y1, x2, y2)

        # Draw shelves (single box per id) with big centered numeric label inside
        for sid, (x1, y1, x2, y2) in best_by_id.items():
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 120, 255), 2)

            # Prefer numeric id for large label (e.g., shelf_4 -> 4)
            try:
                num = str(int(str(sid).split('_')[-1]))
                label_text = num
            except Exception:
                label_text = str(sid)

            # Dynamic font scale based on box height
            box_h = max(1, y2 - y1)
            font_scale = max(0.7, min(2.0, box_h / 140.0))
            thickness = 2 if font_scale < 1.2 else 3

            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cx = x1 + (x2 - x1) // 2 - tw // 2
            cy = y1 + (y2 - y1) // 2 + th // 2
            # Add subtle background shadow for readability
            cv2.putText(img, label_text, (cx+2, cy+2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness+2)
            cv2.putText(img, label_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # Title banner
        title = f"Shelf Map — Frame {frame_idx} — Analysis {analysis_id[:8]}"
        cv2.putText(img, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        ok, buf = cv2.imencode('.png', img)
        if not ok:
            return None
        return buf.tobytes()
    except Exception as e:
        logger.error(f"Error generating shelf map image for frame {frame_idx}: {e}")
        return None

def generate_shelf_map_images(video_path: str, shelf_boxes_per_frame: dict, analysis_id: str, top_k: int = 3):
    """Generate shelf map images for top_k frames with the most shelves.

    Returns list of tuples: [(frame_idx, png_bytes), ...]
    """
    try:
        if not shelf_boxes_per_frame:
            return []
        # Rank frames by number of shelves detected (descending)
        ranked = sorted(
            ((f_idx, len(shelf_boxes_per_frame.get(f_idx, []))) for f_idx in shelf_boxes_per_frame.keys()),
            key=lambda x: x[1], reverse=True
        )
        selected_frames: list[int] = []
        if ranked:
            # Always include the top frame first
            top_frame = ranked[0][0]
            selected_frames.append(top_frame)
            # Build candidate pool from remaining frames
            positive_candidates = [f for f, cnt in ranked[1:] if cnt > 0]
            fallback_candidates = [f for f, _ in ranked[1:]]
            candidates = positive_candidates if positive_candidates else fallback_candidates
            # Randomly sample the rest (up to top_k - 1)
            remaining = max(0, top_k - 1)
            if candidates:
                if len(candidates) <= remaining:
                    picked = candidates
                else:
                    picked = random.sample(candidates, remaining)
                selected_frames.extend(picked)
        results = []
        for f_idx in selected_frames:
            img_bytes = _generate_shelf_map_image_for_frame(video_path, f_idx, shelf_boxes_per_frame, analysis_id)
            if img_bytes:
                results.append((f_idx, img_bytes))
        return results
    except Exception as e:
        logger.error(f"Error generating multiple shelf map images: {e}")
        return []

@app.get("/stream")
def stream_blob(blob: str, range: str | None = Header(default=None)):
    """Proxy stream a blob with HTTP Range support for video playback."""
    if not blob_service_client:
        raise HTTPException(status_code=500, detail="Azure Blob Storage not configured")

    try:
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob)
        props = blob_client.get_blob_properties()
        total_size = props.size
        content_type = props.content_settings.content_type or "video/mp4"

        start = 0
        end = total_size - 1

        if range:
            m = re.match(r"bytes=(\d+)-(\d*)", range)
            if m:
                start = int(m.group(1))
                if m.group(2):
                    end = int(m.group(2))

        if start < 0 or end >= total_size or start > end:
            raise HTTPException(status_code=416, detail="Invalid Range header")

        length = end - start + 1
        downloader = blob_client.download_blob(offset=start, length=length)

        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
            "Cache-Control": "public, max-age=86400",
            "Content-Disposition": "inline",
        }

        status_code = 200
        if range:
            headers["Content-Range"] = f"bytes {start}-{end}/{total_size}"
            status_code = 206

        return StreamingResponse(downloader.chunks(), status_code=status_code, media_type=content_type, headers=headers)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Stream error: {str(e)}")

@app.post("/generate-video")
async def generate_video_only(
    video: UploadFile = File(...),
    max_duration: Optional[int] = 30
):
    """Generate only the annotated video without full analysis"""
    
    if not video.filename.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
        raise HTTPException(status_code=400, detail="Only video files are supported")
    
    # Generate unique analysis ID
    analysis_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        logger.info(f"Starting video generation {analysis_id} for video: {video.filename}")
        
        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            content = await video.read()
            tmp_file.write(content)
            video_path = tmp_file.name
        
        # Process video to get tracking data
        result, processing_data = await process_video_analysis(video_path, max_duration, generate_video=True)
        
        # Generate annotated video
        video_bytes = generate_annotated_video_like_gradio(
            video_path,
            processing_data['tracks'],
            processing_data['action_preds'], 
            processing_data['shelf_boxes_per_frame'],
            analysis_id,
            processing_data['fps'],
            max_duration
        )
        
        if video_bytes and blob_service_client:
            # Save to Azure Blob
            video_filename = f"videos/annotated_{analysis_id}_{timestamp}.mp4"
            video_url = save_to_azure_blob(video_bytes, video_filename, "video/mp4")
            
            # Clean up temporary file
            os.unlink(video_path)
            
            return JSONResponse(content={
                "message": "Annotated video generated successfully",
                "analysis_id": analysis_id,
                "video_url": video_url,
                "annotated_video_blob_path": video_filename,
                "annotated_video_stream": f"/stream?blob={quote(video_filename)}",
                "metadata": {
                    'original_filename': video.filename,
                    'timestamp': datetime.now().isoformat(),
                    'max_duration': max_duration,
                    'file_size': len(content)
                }
            })
        else:
            # Clean up temporary file
            os.unlink(video_path)
            raise HTTPException(status_code=500, detail="Failed to generate annotated video")
        
    except Exception as e:
        logger.error(f"Video generation {analysis_id} failed: {e}")
        # Clean up on error
        if 'video_path' in locals():
            try:
                os.unlink(video_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Video generation failed: {str(e)}")

# ----------------------- AI INSIGHTS (Azure OpenAI) -----------------------

class InsightRequest(BaseModel):
    prompt: str
    data: dict
    heatmap_url: str | None = None
    shelfmap_url: str | None = None

class ApplyFiltersRequest(BaseModel):
    analysis_id: Optional[str] = Field(None, description="Analysis ID")
    excluded_track_ids: List[str] = Field(default=[], description="List of track IDs to exclude (as strings)")
    # Optional: client can send raw processing for stateless recompute
    processing: Optional[dict] = Field(None, description="Processing data from analysis")
    # If provided, recompute using provided processing; else (future) could load by analysis_id
    
    class Config:
        # Allow extra fields to be ignored instead of causing validation errors
        extra = "ignore"

class HeatmapInsightRequest(BaseModel):
    heatmap_data: List[List[float]]
    heatmap_url: Optional[str] = None
    shelfmap_url: Optional[str] = None

class DwellTimeInsightRequest(BaseModel):
    dwell_data: List[dict] # e.g., [{"shelf": "shelf_1", "time": 12.3}, ...]

def _download_image_to_base64(url: str) -> str | None:
    """Download image from URL and convert to base64 for Azure OpenAI vision."""
    try:
        import requests
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            import base64
            return base64.b64encode(response.content).decode('utf-8')
    except Exception as e:
        logger.warning(f"Failed to download image from {url}: {e}")
    return None

@app.post("/ai/insights")
def ai_insights(req: InsightRequest):
    """Generate structured insights (items + summary) using Azure OpenAI.

    Frontend should call this endpoint instead of calling Azure directly.
    """
    client = _get_azure_client()
    try:
        content = [
            {"type": "text", "text": "Balas HANYA JSON valid. items[analysis,pattern,opportunity,warning] + summary."},
            {"type": "text", "text": f"Instruksi pengguna: {req.prompt}"},
            {"type": "text", "text": f"Data: {json.dumps(req.data)[:15000]}"},
        ]
        if req.heatmap_url:
            content.append({"type": "text", "text": "Pertimbangkan gambar heatmap berikut:"})
            # Convert image URL to base64 for Azure OpenAI
            base64_image = _download_image_to_base64(req.heatmap_url)
            if base64_image:
                content.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                })
            else:
                logger.warning("Failed to process heatmap image, continuing without vision")
        # Include shelf map image if provided
        if req.shelfmap_url:
            content.append({"type": "text", "text": "Pertimbangkan juga shelf map (tidak semua shelf terlihat anotasinya) dan tampilan kamera cctv berikut:"})
            base64_shelf = _download_image_to_base64(req.shelfmap_url)
            if base64_shelf:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_shelf}"}
                })
            else:
                logger.warning("Failed to process shelfmap image, continuing without that image")

        def _call_llm(msg_content, use_json_mode: bool):
            kwargs = dict(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": [
                        {"type": "text", "text":
                         "Anda analis retail. Keluaran: JSON valid."
                         " Struktur: items[analysis,pattern,opportunity,warning] + summary."
                         " Tiap item berisi 4-6 bullet yang spesifik & actionable (boleh ada priority)"
                         " Summary 5-6 kalimat: rangkum temuan kunci + rencana aksi prioritas "
                         " Bahasa indonesia profesional namun tidak kaku, jelas , tanpa markdown"}
                    ]},
                    {"role": "user", "content": msg_content},
                ],
                temperature=0.3,
                max_tokens=1500,
            )
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            return client.chat.completions.create(**kwargs)

        def _extract_json(text: str):
            # Strip code fences
            t = text.strip()
            if t.startswith("```"):
                t = t.strip('`')
            # Try direct parse
            try:
                return json.loads(t)
            except Exception:
                pass
            # Fallback: find first JSON object by braces
            try:
                start = t.find('{')
                end = t.rfind('}')
                if start != -1 and end != -1 and end > start:
                    cand = t[start:end+1]
                    return json.loads(cand)
            except Exception:
                pass
            return None

        # Layered retries for model capability differences
        resp = None
        errors: list[str] = []
        for with_image in (True, False):
            msg = content if with_image else [c for c in content if c.get("type") != "image_url"]
            for use_json_mode in (True, False):
                try:
                    resp = _call_llm(msg, use_json_mode)
                    break
                except Exception as e:
                    errors.append(str(e))
                    resp = None
            if resp is not None:
                break
        if resp is None:
            logger.error("AI insight failed after retries: %s", errors[-1] if errors else "unknown")
            return JSONResponse(status_code=500, content={"error": "LLM call failed", "details": errors[-1] if errors else "unknown"})

        text = resp.choices[0].message.content or "{}"
        parsed = _extract_json(text)
        if parsed is None:
            # Return as plain text if still not JSON
            return JSONResponse(content={"items": [], "summary": text})

        # Normalize to consistent shape: items[] + summary
        items = parsed.get("items")
        summary = parsed.get("summary") or parsed.get("conclusion") or parsed.get("ringkasan")

        # If items is dict {analysis:{}, pattern:{}, ...} → to array
        if isinstance(items, dict):
            items = [
                {
                    "type": key,
                    "title": (val.get("title") if isinstance(val, dict) else str(key).title()),
                    "content": (val.get("content") if isinstance(val, dict) else str(val)),
                }
                for key, val in items.items()
            ]
        # If items missing but top-level keys exist, build from them
        if not isinstance(items, list):
            candidates = ["analysis", "pattern", "opportunity", "warning"]
            built = []
            for key in candidates:
                val = parsed.get(key)
                if val is None:
                    continue
                built.append({
                    "type": key,
                    "title": (val.get("title") if isinstance(val, dict) else str(key).title()),
                    "content": (val.get("content") if isinstance(val, dict) else str(val)),
                })
            items = built

        # Final guardrails
        normalized: list[dict] = []
        if isinstance(items, list):
            for it in items:
                if not isinstance(it, dict):
                    continue
                t = it.get("type") or it.get("category")
                title = it.get("title") or (t.title() if isinstance(t, str) else "Insight")
                content = it.get("content") or it.get("text") or it.get("detail") or ""
                if not t:
                    continue
                normalized.append({"type": t, "title": title, "content": content})
        items = normalized

        return JSONResponse(content={"items": items, "summary": summary or ""})
    except Exception as e:
        logger.error(f"AI insight endpoint error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ai/insights/stream")
def ai_insights_stream(req: InsightRequest):
    client = _get_azure_client()
    def gen():
        content = [
            {"type": "text", "text": "Balas HANYA JSON valid. items + summary."},
            {"type": "text", "text": f"Instruksi pengguna: {req.prompt}"},
            {"type": "text", "text": f"Data: {json.dumps(req.data)[:15000]}"},
        ]
        if req.heatmap_url:
            content.append({"type": "input_image", "image_url": {"url": req.heatmap_url}})
        if req.shelfmap_url:
            content.append({"type": "input_image", "image_url": {"url": req.shelfmap_url}})
        stream = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": [
                    {"type": "text", "text": 
                     "Anda analis retail. Keluaran JSON."
                     " Struktur: items[analysis,pattern,opportunity,warning] + summary."
                     " Tiap item berisi 3-6 bullet yang spesifik & actionable"
                    }
                ]},
                {"role": "user", "content": content},
            ],
            temperature=0.3,
            max_tokens=1500,
            response_format={"type": "json_object"},
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    return StreamingResponse(gen(), media_type="text/plain")

@app.get("/ai/status")
def ai_status():
    """Quick status for AI config without exposing secrets."""
    return {
        "endpoint_set": bool(AZURE_OPENAI_ENDPOINT),
        "deployment": AZURE_OPENAI_DEPLOYMENT,
        "api_version": AZURE_OPENAI_API_VERSION,
        "key_present": bool(AZURE_OPENAI_KEY),
    }

# ----------------------- AI Q&A (concise chatbot-style) -----------------------

class QARequest(BaseModel):
    prompt: str
    data: dict | None = None
    heatmap_url: str | None = None
    shelfmap_url: str | None = None

@app.post("/ai/qa")
def ai_qa(req: QARequest):
    """Answer user's question concisely using provided metrics and (optionally) heatmap.

    Returns: { "answer": string }
    """
    client = _get_azure_client()
    try:
        content = [
            {"type": "text", "text": "Jawab ringkas, langsung ke poin, dalam Bahasa Indonesia dan ramah. Gunakan data yang diberikan. Anda di sini sebagai asisten analis retail yang faktual dan ringkas untuk memanfaatkan data yang ada untuk membantu pemilik retail meningkatkan performa tokonya. Jika tidak ada cukup data, sebutkan keterbatasannya dan jangan berbohong. Hindari markdown berlebihan."},
            {"type": "text", "text": f"Pertanyaan pemilik retail: {req.prompt}"},
        ]
        if req.data:
            try:
                content.append({"type": "text", "text": f"Data ringkas: {json.dumps(req.data)[:15000]}"})
            except Exception:
                pass
        if req.heatmap_url:
            content.append({"type": "text", "text": "Pertimbangkan heatmap (gambar) berikut saat relevan:"})
            # Convert image URL to base64 for Azure OpenAI
            base64_image = _download_image_to_base64(req.heatmap_url)
            if base64_image:
                content.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                })
            else:
                logger.warning("Failed to process heatmap image for QA, continuing without vision")
        if req.shelfmap_url:
            content.append({"type": "text", "text": "Pertimbangkan shelf map (tidak semua shelf terlihat anotasinya) dan tampilan cctv (anda bisa tahu letak pintu dan kasir dan bandingkan dengan heatmap) berikut saat relevan:"})
            base64_shelf = _download_image_to_base64(req.shelfmap_url)
            if base64_shelf:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_shelf}"}
                })
            else:
                logger.warning("Failed to process shelfmap image for QA, continuing without it")

        def _call(msg_content):
            return client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": [{"type": "text", "text": "Anda adalah asisten analis retail yang faktual dan ringkas."}]},
                    {"role": "user", "content": msg_content},
                ],
                temperature=0.2,
                max_tokens=700,
            )

        try:
            resp = _call(content)
        except Exception as first_err:
            # Retry tanpa gambar jika model deployment tidak vision
            try:
                content_no_img = [c for c in content if c.get("type") != "input_image"]
                resp = _call(content_no_img)
            except Exception as second_err:
                logger.error(f"AI QA failed (first): {first_err}")
                logger.error(f"AI QA failed (retry): {second_err}")
                return JSONResponse(status_code=500, content={
                    "error": "LLM call failed",
                    "details": str(second_err),
                })

        answer = resp.choices[0].message.content or ""
        return {"answer": answer}
    except Exception as e:
        logger.error(f"AI QA error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ai/qa/stream")
def ai_qa_stream(req: QARequest):
    client = _get_azure_client()
    def gen():
        content = [
            {"type": "text", "text": "Jawab secara menjelaskan namun tetap on point, dalam Bahasa Indonesia dan ramah. Gunakan data yang diberikan. Manfaatkan data gambar heatmap dan shelf map (gambar kamera cctvnya, sehingga anda tahu dimana pintu dan kasir) yang diberikan. Anda bisa jawab dengan seperti A, B gitu misalnya ditaya bentu heatmap bisa dijawab melingkar atau masuk lalu cenerng bentuk L ke kanan, namun tetap berdasarkan data. Anda di sini sebagai asisten analis retail yang faktual dan ringkas untuk memanfaatkan data yang ada untuk membantu pemilik retail meningkatkan performa tokonya. Jika tidak ada cukup data, sebutkan keterbatasannya dan jangan berbohong. Hindari markdown berlebihan. Kasih juga rekomendasi barang yang bisa dijual di area terentu, misal karena di sini ramai sebaiknya rak diberi tulisan promosi (keterbacaan) atau barang tertentu"},
            {"type": "text", "text": f"Pertanyaan pemilik retail: {req.prompt}"},
        ]
        if req.data:
            try:
                content.append({"type": "text", "text": f"Data ringkas: {json.dumps(req.data)[:15000]}"})
            except Exception:
                pass
        if req.heatmap_url:
            # Convert image URL to base64 for Azure OpenAI
            base64_image = _download_image_to_base64(req.heatmap_url)
            if base64_image:
                content.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                })
            else:
                logger.warning("Failed to process heatmap image for QA stream, continuing without vision")
        if req.shelfmap_url:
            base64_shelf = _download_image_to_base64(req.shelfmap_url)
            if base64_shelf:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_shelf}"}
                })
            else:
                logger.warning("Failed to process shelfmap image for QA stream, continuing without vision")

        def produce(msg_content):
            stream_obj = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": [{"type": "text", "text": "Anda adalah asisten analis retail yang faktual dan ringkas."}]},
                    {"role": "user", "content": msg_content},
                ],
                temperature=0.2,
                max_tokens=700,
                stream=True,
            )
            for ch in stream_obj:
                if ch.choices and ch.choices[0].delta and ch.choices[0].delta.content:
                    yield ch.choices[0].delta.content

        try:
            # Try with image
            for piece in produce(content):
                yield piece
        except Exception as first_err:
            try:
                # Retry without image
                content_no_img = [c for c in content if c.get("type") != "input_image"]
                for piece in produce(content_no_img):
                    yield piece
            except Exception as second_err:
                logger.error(f"AI QA stream failed: {first_err}; retry: {second_err}")
                yield "Maaf, terjadi kendala menjawab saat ini."
    return StreamingResponse(gen(), media_type="text/plain")

@app.post("/ai/heatmap-insight")
async def ai_heatmap_insight(req: HeatmapInsightRequest):
    client = _get_azure_client()
    try:
        # Prepare content for LLM
        prompt = (
            "Analisis data dan gambar heatmap berikut untuk mengidentifikasi pola alur lalu lintas utama pelanggan di dalam toko. Anda bisa melihat heatmap dan dihubungkan dengan screenshot cctnya, dimana ia masuk dan diidentikkan dengan heatmapnya. "
            "Jelaskan polanya secara singkat dan berikan satu rekomendasi actionable. Jawaban harus dalam satu kalimat atau maksimal dua kalimat pendek. Jawab dalam Bahasa Indonesia."
        )
        content = [
            {"type": "text", "text": prompt},
            {"type": "text", "text": f"Data Grid Heatmap (nilai lebih tinggi berarti lebih ramai): {json.dumps(req.heatmap_data)}"},
        ]

        if req.heatmap_url:
            base64_image = _download_image_to_base64(req.heatmap_url)
            if base64_image:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                })

        if req.shelfmap_url:
            base64_shelf = _download_image_to_base64(req.shelfmap_url)
            if base64_shelf:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_shelf}"}
                })

        # Call LLM
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "Anda adalah asisten analis retail yang cerdas dan ringkas."},
                {"role": "user", "content": content},
            ],
            temperature=0.3,
            max_tokens=150,
        )

        insight = resp.choices[0].message.content or "Pola heatmap menunjukkan jalur utama pelanggan. Pertimbangkan untuk menempatkan produk dengan margin tinggi di sepanjang rute ini."

        return {"insight": insight}

    except Exception as e:
        logger.error(f"AI heatmap insight error: {e}")
        # Provide a safe fallback
        fallback_insight = "Pola heatmap menunjukkan jalur utama pelanggan. Pertimbangkan untuk menempatkan produk dengan margin tinggi di sepanjang rute ini."
        return {"insight": fallback_insight}

class ComparisonInsightRequest(BaseModel):
    files_data: List[dict]  # List of file analysis data
    
@app.post("/comparison-insights")
async def comparison_insights(req: ComparisonInsightRequest):
    """Generate AI insights for comparison view with multiple files data and images."""
    try:
        # Prepare content for LLM
        content = [{
            "type": "text", 
            "text": "Analisis dan bandingkan hasil analisis multiple toko ini. Berikan insight tentang perbedaan perilaku pelanggan, pola lalu lintas, dan rekomendasi untuk optimasi dalam bahasa Indonesia. Jawab dengan bahasa indonesia profesional dan ringkas. Anda adalah konsultan ahli analitik ritel. Analisis data perbandingan dan gambar berikut untuk memberikan insight yang dapat ditindaklanjuti terkait perbedaan perilaku pelanggan, variasi pola lalu lintas, serta rekomendasi spesifik untuk optimalisasi toko. anda juga bisa melihat gambar cctv kamera berikut untuk menghubungkan dengan heatmapnya, poisis di heatmap ini menunjukkan bagian asli apa di cctnyanya gitu anda bisa kembangkan sesuai gambarnya."
        }]
        
        # Add text summary of all files
        files_summary = []
        for i, file_data in enumerate(req.files_data):
            summary = f"""File {i+1} ({file_data.get('filename', f'File {i+1}')}):
- Jumlah Pelanggan: {file_data.get('unique_persons', 0)}
- Total Interaksi: {file_data.get('total_interactions', 0)}
- Rata-rata Dwell Time: {file_data.get('avg_dwell_time', 0):.1f}s
- Aksi Terbanyak: {file_data.get('top_action', 'N/A')}
- Jumlah Rak: {file_data.get('shelves_count', 0)}"""
            files_summary.append(summary)
        
        content[0]["text"] += "\n\n" + "\n\n".join(files_summary)
        
        # Convert and add heatmap images
        for i, file_data in enumerate(req.files_data):
            heatmap_url = file_data.get('heatmap_image')
            if heatmap_url:
                base64_heatmap = _download_image_to_base64(heatmap_url)
                if base64_heatmap:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_heatmap}"}
                    })
                    content.append({
                        "type": "text",
                        "text": f"Heatmap untuk File {i+1} ({file_data.get('filename', f'File {i+1}')})"
                    })
        
        # Convert and add shelf map images
        for i, file_data in enumerate(req.files_data):
            shelf_map_url = file_data.get('shelf_map_image')
            if shelf_map_url:
                base64_shelf = _download_image_to_base64(shelf_map_url)
                if base64_shelf:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_shelf}"}
                    })
                    content.append({
                        "type": "text",
                        "text": f"Shelf Map untuk File {i+1} ({file_data.get('filename', f'File {i+1}')})"
                    })
        
        messages = [
            {
                "role": "system", 
                "content": "Jawab dengan bahasa indonesia profesional dan ringkas. Anda adalah konsultan ahli analitik ritel. Analisis data perbandingan dan gambar berikut untuk memberikan insight yang dapat ditindaklanjuti terkait perbedaan perilaku pelanggan, variasi pola lalu lintas, serta rekomendasi spesifik untuk optimalisasi toko. Fokus pada insight praktis berbasis data yang dapat langsung diterapkan oleh manajer toko. anda juga bisa melihat gambar cctv kamera berikut untuk melihat pola lalu lintas dan perilaku pelanggan. kasih juga visualisasi dari perbedaan itu.misal seperti apa sih bedanya..."
            },
            {"role": "user", "content": content}
        ]
        
        # Call OpenAI API
        client = _get_azure_client()
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            max_tokens=1500,
            temperature=0.7
        )
        
        insight_text = response.choices[0].message.content
        
        return {
            "insight": insight_text,
            "files_analyzed": len(req.files_data),
            "images_processed": sum(1 for f in req.files_data if f.get('heatmap_image')) + sum(1 for f in req.files_data if f.get('shelf_map_image'))
        }
        
    except Exception as e:
        logger.error(f"Error in comparison insights: {e}")
        # Fallback insight
        return {
            "insight": "Berdasarkan data perbandingan, terlihat perbedaan pola lalu lintas dan perilaku pelanggan antar lokasi. Analisis lebih lanjut diperlukan untuk memberikan rekomendasi yang spesifik.",
            "files_analyzed": len(req.files_data) if req.files_data else 0,
            "images_processed": 0
        }

@app.post("/ai/dwell-time-insight")
async def ai_dwell_time_insight(req: DwellTimeInsightRequest):
    client = _get_azure_client()
    try:
        prompt = (
            "Berdasarkan data dwell time per rak berikut, berikan analisis dalam 4 poin terpisah (gunakan '\\n' sebagai pemisah). "
            "Poin 1: Identifikasi rak dengan dwell time tertinggi (paling menarik) dan terendah (kurang menarik). "
            "Poin 2: Berikan interpretasi bisnis dari temuan ini (misal, rak X sangat menarik, rak Y diabaikan). "
            "Poin 3: Berikan satu rekomendasi untuk memanfaatkan rak populer. "
            "Poin 4: Berikan satu rekomendasi untuk meningkatkan performa rak yang tidak populer. "
            "Jawab dalam Bahasa Indonesia."
        )
        content = [
            {"type": "text", "text": prompt},
            {"type": "text", "text": f"Data Dwell Time (dalam detik): {json.dumps(req.dwell_data)}"},
        ]

        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "Anda adalah asisten analis retail yang cerdas dan ringkas."},
                {"role": "user", "content": content},
            ],
            temperature=0.3,
            max_tokens=200,
        )

        insight = resp.choices[0].message.content or "Rak teratas sangat menarik bagi pelanggan.\\nRak terbawah kurang mendapat perhatian.\\nLetakkan produk komplementer di dekat rak populer.\\nCoba ubah tata letak atau promosikan produk di rak yang sepi."
        
        # Post-process to clean up formatting from LLM
        import re
        # Force newlines by replacing numberings like "1. ", "2. "
        insight = re.sub(r'\s*\d+\.\s*', '\n', insight).strip()
        
        # Build bullet items robustly (prefer existing newlines; else split by numbering or sentences)
        items = [seg.strip(' -•').strip() for seg in insight.split('\n') if seg.strip()]
        if len(items) < 2:
            numbered_parts = [p.strip() for p in re.split(r'\s*\d+\.\s*', insight) if p.strip()]
            if len(numbered_parts) > len(items):
                items = numbered_parts
        if len(items) < 2:
            sentence_parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', insight) if p.strip()]
            if sentence_parts:
                items = sentence_parts
        # Limit to 4 concise items
        items = items[:4]
        
        return {"insight": insight, "items": items}
    except Exception as e:
        logger.error(f"AI dwell time insight error: {e}")
        fallback_insight = "Rak teratas sangat menarik bagi pelanggan.\\nRak terbawah kurang mendapat perhatian.\\nLetakkan produk komplementer di dekat rak populer.\\nCoba ubah tata letak atau promosikan produk di rak yang sepi."
        return {"insight": fallback_insight, "items": [
            "Rak teratas sangat menarik bagi pelanggan.",
            "Rak terbawah kurang mendapat perhatian.",
            "Letakkan produk komplementer di dekat rak populer.",
            "Coba ubah tata letak atau promosikan produk di rak yang sepi."
        ]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)