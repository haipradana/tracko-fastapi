# Retail Behavior Analysis FastAPI

FastAPI-based retail customer behavior analysis with ML models and Azure Blob Storage integration.

## 🚀 Features

- **Video Upload & Processing**: Upload retail surveillance videos
- **Real-time Analysis**: AI-powered customer behavior analysis
- **Azure Blob Storage**: Cloud storage for results, heatmaps, and reports
- **Multiple Models**: YOLO person detection, shelf segmentation, action classification
- **Heatmap Generation**: Customer traffic visualization
- **CSV Reports**: Detailed analytics export
- **RESTful API**: Easy integration with web applications

## 🛠 Technology Stack

- **FastAPI** - Modern Python web framework
- **PyTorch** - Deep learning framework
- **Ultralytics YOLO** - Object detection
- **Transformers** - Action classification
- **Azure Blob Storage** - Cloud storage
- **Docker** - Containerization

## 📋 Prerequisites

- Python 3.9+
- Azure Storage Account
- Docker (optional)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo>
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

### 4. Configure Azure Blob Storage

Copy environment file and configure:

```bash
cp env.example .env
```

Edit `.env`:
```env
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=yourstorageaccount;AccountKey=yourstoragekey;EndpointSuffix=core.windows.net
AZURE_STORAGE_CONTAINER=retail-analysis-results
```

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

#### 3. Memory Issues
```bash
# Increase swap space
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 4. Port Already in Use
```bash
# Change port
uvicorn main:app --host 0.0.0.0 --port 8001
```
## 🚀 Production Deployment

### 1. Azure VM Deployment
```bash
# Upload to VM
scp -r fastapi-app/ azureuser@your-vm-ip:/home/azureuser/

# Setup on VM
ssh azureuser@your-vm-ip
cd fastapi-app
python scripts/download_models.py
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. Docker Production
```bash
# Build and run
docker build -t retail-api .
docker run -p 8000:8000 --env-file .env retail-api
```

### 3. Systemd Service
```bash
# Create service file
sudo nano /etc/systemd/system/retail-api.service

[Unit]
Description=Retail Behavior Analysis API
After=network.target

[Service]
Type=simple
User=azureuser
WorkingDirectory=/home/azureuser/fastapi-app
Environment=PATH=/home/azureuser/fastapi-app/venv/bin
ExecStart=/home/azureuser/fastapi-app/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target

# Enable and start
sudo systemctl enable retail-api
sudo systemctl start retail-api
sudo systemctl restart retail-api

```

## 📞 Support

For issues and questions:
- Check logs: `tail -f logs/app.log`
- Test health: `curl http://localhost:8000/health`
- Check Azure Blob: Verify connection string and container

## 📄 License

This project is licensed under the MIT License. 