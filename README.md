# PP-Structure V3 Demo

A Streamlit web application for document layout analysis using PaddleOCR's PP-Structure V3.

## Features

- Document layout detection and analysis
- Text detection and recognition
- Memory management and optimization
- Web-based interface for easy use
- **Model download from Kaggle Hub** (NEW)

## Model Management

### Using Kaggle Hub Models

The application now supports downloading models directly from Kaggle Hub. This provides an alternative to manually managing model files.

#### Setup Kaggle Hub

1. Install kagglehub (already included in requirements.txt):
   ```bash
   pip install kagglehub
   ```

2. Configure Kaggle authentication (optional, for private models):
   ```bash
   # Set up Kaggle API credentials if needed
   export KAGGLE_USERNAME=your_username
   export KAGGLE_KEY=your_api_key
   ```

#### Using Kaggle Models in the App

1. Start the application
2. Click the "Download Models from Kaggle" button in the Model Management section
3. Wait for the download to complete
4. The application will automatically use the downloaded models

#### Current Kaggle Model

- **Repository**: `phuchoangnguyen/model_paddle_layout_nhom_nhan/pyTorch/default`
- **Type**: PP-Structure V3 compatible models
- **Components**: Layout detection, text detection, and text recognition models

### Local Model Files

### Prerequisites

- Python 3.10+
- Git

### Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run web_demo.py
   ```

## Docker Deployment

### Build and Run Locally

```bash
# Build the Docker image
docker build -t paddle-layout-demo .

# Run the container
docker run -p 8501:8501 paddle-layout-demo
```

## Railway Deployment

### Quick Deploy

1. Fork/clone this repository
2. Connect your repository to Railway
3. Railway will automatically detect the Dockerfile and deploy

### Manual Deploy

1. Install Railway CLI:
   ```bash
   npm install -g @railway/cli
   ```

2. Login to Railway:
   ```bash
   railway login
   ```

3. Deploy:
   ```bash
   railway up
   ```

### Environment Variables

Railway will automatically set the `PORT` environment variable. No additional configuration is needed.

### Configuration Files

- `Dockerfile`: Docker configuration for containerizing the application
- `railway.json`: Railway-specific deployment configuration
- `.dockerignore`: Files to ignore during Docker build
- `start.sh`: Railway startup script
- `requirements.txt`: Python dependencies

## Model Files

The application supports two ways to provide models:

### Option 1: Kaggle Hub Models (Recommended)
Use the "Download Models from Kaggle" button in the web interface to automatically download the latest models.

### Option 2: Local Model Files
Manually place model files in the following structure:

```
models/
├── layout_detection/
│   ├── inference.json
│   ├── inference.pdiparams
│   └── inference.yml
├── text_detection/
│   ├── inference.json
│   ├── inference.pdiparams
│   └── inference.yml
└── text_recognition/
    ├── inference.json
    ├── inference.pdiparams
    └── inference.yml
```

**Note**: If both Kaggle and local models are available, the application will prefer Kaggle models.

## Memory Management

The application includes comprehensive memory management features:

- Automatic cache clearing
- Memory usage monitoring
- Resource cleanup after processing
- Optimized for deployment environments

## Usage

1. Upload an image (JPG, JPEG, or PNG)
2. Wait for processing to complete
3. View the original image and detection results
4. Download JSON results if needed

## Troubleshooting

### Memory Issues
- Use the "Clear All Cache" button in the sidebar
- The application automatically manages memory between requests

### Model Loading Issues
- **Kaggle Models**: Click "Download Models from Kaggle" and ensure you have internet connectivity
- **Local Models**: Ensure all model files are present in the `models/` directory
- Check that model files are not corrupted
- Verify the kagglehub package is installed: `pip install kagglehub`

### Kaggle Hub Issues
- **Download Failed**: Check your internet connection and verify the Kaggle model repository exists
- **Authentication Error**: For private models, ensure Kaggle API credentials are configured
- **Permission Error**: Make sure the application has write permissions to download models

### Deployment Issues
- Verify all dependencies are listed in `requirements.txt`
- Check Railway logs for specific error messages
- Ensure kagglehub is included in requirements.txt for cloud deployments
