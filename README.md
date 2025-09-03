# FastAPI Backend for TRACKO

Backend infrastructure for Tracko.tech - A comprehensive retail analytics platform powered by FastAPI, featuring real-time customer behavior analysis, computer vision processing, and seamless cloud storage integration.

## Features

- **Video Processing API**: Process and analyze retail CCTV video streams
- **Customer Analytics Engine**: Real-time AI analysis of customer behavior patterns
- **Storage Integration**: Flexible storage system supporting both Azure Blob and local storage
- **Computer Vision Pipeline**: 
  - **Person Detection and Tracking**: YOLO + BotSORT + ReID for accurate person detection and tracking
  - **Shelf Interaction Analysis**: YOLO + Shelf Segmentation for identifying customer interactions
  - **Customer Action Classification**: VideoMAE  model for classifying customer actions
- **Analytics Visualization**: Generate heatmaps showing customer movement patterns
- **Data Export API**: Export analyzed data in CSV format for further processing
- **RESTful Endpoints**: Documented API endpoints for frontend integration

## Technology Stack

- **FastAPI** - Modern Python web framework
- **PyTorch** - Deep learning framework
- **Ultralytics YOLO** - Object detection and Shelf Segmentation
- **VideoMAE** - Action classification
- **Azure Blob Storage** - Cloud storage
- **Docker** - Containerization

## Prerequisites

- Python 3.9+
- Azure Storage Account (or Local Storage)
- Docker (optional)

## Installation

### 1. Clone and Setup

```bash
git clone https://github.com/haipradana/tracko-fastapi.git
cd fastapi-app
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Models

```bash
python scripts/download_models.py
```

#### Step 4a: Configure Azure Blob Storage (Optional)

If you want to use Azure Blob Storage for cloud storage:

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your Azure configurations
notepad .env  # Windows
# nano .env    # Linux/Mac
```

Edit `.env`:
```env
AZURE_STORAGE_CONNECTION_STRING=<your-connection-string>
AZURE_STORAGE_CONTAINER=<your-container-name>

# Azure OpenAI (Optional - for AI insights)
AZURE_OPENAI_ENDPOINT=<your-endpoint>
AZURE_OPENAI_KEY=<your-openai-key>
AZURE_OPENAI_DEPLOYMENT=gpt-4
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

#### Step 4b: Use Local Storage (Recommended for Local Development)

If you want to use local storage (no Azure configuration needed):

```bash
# Option 1: Skip .env file creation entirely
# The system will automatically use local storage

# Option 2: Create empty .env file
touch .env  # Linux/Mac
type nul > .env  # Windows

# Option 3: Create .env with local storage settings
echo "# Using local storage - no Azure configuration needed" > .env
echo "AZURE_STORAGE_CONNECTION_STRING=" >> .env
echo "AZURE_STORAGE_CONTAINER=retail-analysis-results" >> .env
```

**Note**: With local storage, all results (heatmaps, CSV reports, analysis data) will be saved in local directories and served as base64 data URIs.

### 5. Run FastAPI

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Docker (Alternative)

```bash
docker-compose up -d
```

##  Troubleshooting

### Common Issues

#### 1. Model Download Failed
```bash
# Manual download
python scripts/download_models.py
```

#### 2. Azure Blob Storage Error
```bash
# Check connection string
echo $AZURE_STORAGE_CONNECTION_STRING

# Test connection
python -c "from azure.storage.blob import BlobServiceClient; print('Connection OK')"
```

#### 3. Port Already in Use
```bash
# Change port
uvicorn main:app --host 0.0.0.0 --port 8001
```

## License

This project is licensed under the Apache License 2.0.