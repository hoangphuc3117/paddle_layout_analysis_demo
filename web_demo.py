
import streamlit as st
import gc
import os
import psutil
import hashlib
import requests
import zipfile
from pathlib import Path

# Set OpenCV environment variables for headless operation
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['OPENCV_IO_ENABLE_JASPER'] = '1' 
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Set page config for fullscreen (wide) layout
st.set_page_config(page_title="PP-Structure V3 Demo", layout="wide")
# from paddlex import create_model
from paddleocr import PPStructureV3
import os
import shutil
import glob
import json
import tempfile

st.title("PP-Structure V3 Demo")

# Model download URLs - Replace these with your actual hosted model URLs
MODEL_URLS = {
    "layout_detection": {
        "inference.pdiparams": "https://your-host.com/models/layout_detection/inference.pdiparams",
        "inference.json": "https://your-host.com/models/layout_detection/inference.json", 
        "inference.yml": "https://your-host.com/models/layout_detection/inference.yml"
    },
    "text_detection": {
        "inference.pdiparams": "https://your-host.com/models/text_detection/inference.pdiparams",
        "inference.json": "https://your-host.com/models/text_detection/inference.json",
        "inference.yml": "https://your-host.com/models/text_detection/inference.yml"
    },
    "text_recognition": {
        "inference.pdiparams": "https://your-host.com/models/text_recognition/inference.pdiparams", 
        "inference.json": "https://your-host.com/models/text_recognition/inference.json",
        "inference.yml": "https://your-host.com/models/text_recognition/inference.yml"
    }
}

def download_file_with_progress(url, local_path, description="Downloading"):
    """Download a file with progress bar using requests"""
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Download with requests
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(local_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = min(100, (downloaded * 100) // total_size)
                        progress_bar.progress(percent / 100)
                        status_text.text(f"{description}: {percent}% ({downloaded//1024//1024}MB/{total_size//1024//1024}MB)")
        
        progress_bar.empty()
        status_text.empty()
        return True
    except Exception as e:
        st.error(f"Failed to download {url}: {str(e)}")
        return False

def check_and_download_models():
    """Check if models exist, download if missing"""
    models_dir = Path("models")
    
    # Check if any model files are missing or are LFS pointer files (< 1KB indicates LFS pointer)
    missing_files = []
    
    for model_type, files in MODEL_URLS.items():
        model_path = models_dir / model_type
        for filename, url in files.items():
            file_path = model_path / filename
            if not file_path.exists() or file_path.stat().st_size < 1024:  # LFS pointer files are tiny
                missing_files.append((model_type, filename, url, file_path))
    
    if missing_files:
        st.warning(f"Found {len(missing_files)} model files to download. This may take a few minutes...")
        
        # Create a container for download progress
        download_container = st.container()
        
        with download_container:
            for model_type, filename, url, file_path in missing_files:
                st.info(f"Downloading {model_type}/{filename}...")
                
                if download_file_with_progress(url, str(file_path), f"{model_type}/{filename}"):
                    st.success(f"✅ Downloaded {model_type}/{filename}")
                else:
                    st.error(f"❌ Failed to download {model_type}/{filename}")
                    return False
        
        st.success("🎉 All model files downloaded successfully!")
        return True
    else:
        st.success("✅ All model files are already available!")
        return True

def get_github_lfs_url(repo_owner, repo_name, file_path, branch="main"):
    """Generate GitHub LFS download URL"""
    return f"https://github.com/{repo_owner}/{repo_name}/raw/{branch}/{file_path}"

def setup_models_from_github_lfs():
    """Alternative: Download directly from GitHub LFS"""
    repo_owner = "hoangphuc3117"  
    repo_name = "paddle_layout_analysis_demo"
    
    models_to_download = [
        "models/layout_detection/inference.pdiparams",
        "models/layout_detection/inference.json", 
        "models/layout_detection/inference.yml",
        "models/text_detection/inference.pdiparams",
        "models/text_detection/inference.json",
        "models/text_detection/inference.yml", 
        "models/text_recognition/inference.pdiparams",
        "models/text_recognition/inference.json",
        "models/text_recognition/inference.yml"
    ]
    
    st.info("Downloading model files from GitHub LFS...")
    
    for file_path in models_to_download:
        local_path = Path(file_path)
        
        # Skip if file exists and is not an LFS pointer
        if local_path.exists() and local_path.stat().st_size > 1024:
            continue
            
        url = get_github_lfs_url(repo_owner, repo_name, file_path)
        
        if download_file_with_progress(url, str(local_path), f"Downloading {local_path.name}"):
            st.success(f"✅ Downloaded {file_path}")
        else:
            st.error(f"❌ Failed to download {file_path}")
            return False
    
    return True

# Initialize session state
if 'current_file_id' not in st.session_state:
    st.session_state.current_file_id = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'temp_file_path' not in st.session_state:
    st.session_state.temp_file_path = None
if 'memory_stats' not in st.session_state:
    st.session_state.memory_stats = {}
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

def get_file_id(uploaded_file):
    """Generate unique ID for uploaded file"""
    if uploaded_file is None:
        return None
    
    # Create hash from file content and metadata
    file_content = uploaded_file.read()
    uploaded_file.seek(0)  # Reset file pointer
    
    file_info = f"{uploaded_file.name}_{uploaded_file.size}_{len(file_content)}"
    return hashlib.md5(file_info.encode()).hexdigest()

# Load model once
@st.cache_resource
def load_model():
    # First, ensure all model files are downloaded
    if not check_and_download_models():
        # Fallback to GitHub LFS if custom URLs don't work
        if not setup_models_from_github_lfs():
            st.error("Failed to download model files. Please check your internet connection.")
            st.stop()
    
    # model_name = "PP-DocLayout_plus-L"
    # model_dir = "model"
    # model = create_model(model_name=model_name, model_dir=model_dir, device="cpu")
    model = PPStructureV3(
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        layout_detection_model_name="PP-DocLayout-L",
        layout_detection_model_dir="models/layout_detection",
        text_detection_model_name="PP-OCRv5_server_det",
        text_detection_model_dir="models/text_detection",
        text_recognition_model_name="PP-OCRv5_server_rec",
        text_recognition_model_dir="models/text_recognition",
        use_table_recognition=False,
        use_seal_recognition=False,
        use_chart_recognition=False,
        use_formula_recognition=False,
    )
    return model

model = load_model()

# Add manual download option in sidebar
with st.sidebar:
    st.header("Model Management")
    
    if st.button("🔄 Re-download Models"):
        st.cache_resource.clear()
        # Force re-download by removing existing files
        shutil.rmtree("models", ignore_errors=True)
        st.rerun()
    
    st.info("""
    **Note for Railway Deployment:**
    - Model files are automatically downloaded on first run
    - This may take 2-3 minutes initially  
    - Files are cached between deployments
    """)

uploaded_file = st.file_uploader("Upload an image for layout inference", type=["jpg", "jpeg", "png"])

# Check if file has changed
current_file_id = get_file_id(uploaded_file)
    
# If file has changed, clean up previous results and process new file
if current_file_id != st.session_state.current_file_id:

    # Update current file ID
    st.session_state.current_file_id = current_file_id
    
    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        st.session_state.temp_file_path = tmp_file.name

    # Run prediction with loading spinner
    with st.spinner('Processing... Please wait while we analyze your image.'):
        print(f"Processing image: {st.session_state.temp_file_path}")
        try:
            output = model.predict(st.session_state.temp_file_path, batch_size=1)
        except Exception as e:
            st.error(f"Error during prediction: {e}")

        output_dir = "output"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.makedirs(output_dir, exist_ok=True)

        # Save results
        if output:
            for res in output:
                res.save_to_img(output_dir)
                res.save_to_json(output_dir)

        # Store results in session state
        st.session_state.results = {
            'output_dir': output_dir,
            'image_files': glob.glob(os.path.join(output_dir, "*.jpg")) + \
                          glob.glob(os.path.join(output_dir, "*.jpeg")) + \
                          glob.glob(os.path.join(output_dir, "*.png")),
            'json_files': glob.glob(os.path.join(output_dir, "*.json"))
        }

# Display results if available
if st.session_state.results and st.session_state.temp_file_path:
    st.subheader("Original Image")
    st.image(st.session_state.temp_file_path, caption="Uploaded Image", use_container_width=False)
    
    st.subheader("Detection Results")
    
    image_files = st.session_state.results['image_files']
    json_files = st.session_state.results['json_files']
    
    # Display images
    if image_files:
        for i in range(0, len(image_files), 2):
            col1, col2 = st.columns(2)
            
            # First image in left column
            img_path1 = sorted(image_files)[i]
            filename1 = os.path.basename(img_path1)
            with col1:
                st.image(img_path1, caption=f"Result: {filename1}", use_container_width=False)
            
            # Second image in right column (if exists)
            if i + 1 < len(image_files):
                img_path2 = sorted(image_files)[i + 1]
                filename2 = os.path.basename(img_path2)
                with col2:
                    st.image(img_path2, caption=f"Result: {filename2}", use_container_width=False)
    
    # Display JSON results
    if json_files:
        st.subheader("Detection Results (JSON)")
        for json_path in sorted(json_files):
            filename = os.path.basename(json_path)
            with open(json_path, "r") as f:
                json_data = json.load(f)
            st.text(f"File: {filename}")
            st.json(json_data, expanded=False)