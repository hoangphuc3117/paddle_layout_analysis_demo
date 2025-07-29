
import streamlit as st
import gc
import os
import psutil
import hashlib
import kagglehub

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

def get_file_id(uploaded_file):
    """Generate unique ID for uploaded file"""
    if uploaded_file is None:
        return None
    
    # Create hash from file content and metadata
    file_content = uploaded_file.read()
    uploaded_file.seek(0)  # Reset file pointer
    
    file_info = f"{uploaded_file.name}_{uploaded_file.size}_{len(file_content)}"
    return hashlib.md5(file_info.encode()).hexdigest()

def download_models_from_kaggle():
    """Download models from Kaggle Hub"""
    try:
        with st.spinner("Downloading models from Kaggle Hub... This may take a few minutes."):
            # Download latest version
            path = kagglehub.model_download("phuchoangnguyen/model_paddle_layout_nhom_nhan/pyTorch/default")
        return path
    except Exception as e:
        return None

# Load model once
@st.cache_resource
def load_model():
    st.session_state.kaggle_model_path = download_models_from_kaggle()
    # Check if Kaggle models are available
    kaggle_model_path = getattr(st.session_state, 'kaggle_model_path', None)
    
    if kaggle_model_path and os.path.exists(kaggle_model_path):
        # You may need to adjust these paths based on the structure of the downloaded models
        layout_detection_dir = os.path.join(kaggle_model_path, "models/layout_detection") if os.path.exists(os.path.join(kaggle_model_path, "models/layout_detection")) else "models/layout_detection"
        text_detection_dir = os.path.join(kaggle_model_path, "models/text_detection") if os.path.exists(os.path.join(kaggle_model_path, "models/text_detection")) else "models/text_detection"
        text_recognition_dir = os.path.join(kaggle_model_path, "models/text_recognition") if os.path.exists(os.path.join(kaggle_model_path, "models/text_recognition")) else "models/text_recognition"
    else:
        # Use default local models
        layout_detection_dir = "models/layout_detection"
        text_detection_dir = "models/text_detection"
        text_recognition_dir = "models/text_recognition"
    
    # model_name = "PP-DocLayout_plus-L"
    # model_dir = "model"
    # model = create_model(model_name=model_name, model_dir=model_dir, device="cpu")
    model = PPStructureV3(
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        layout_detection_model_name="PP-DocLayout-L",
        layout_detection_model_dir=layout_detection_dir,
        text_detection_model_name="PP-OCRv5_server_det",
        text_detection_model_dir=text_detection_dir,
        text_recognition_model_name="PP-OCRv5_server_rec",
        text_recognition_model_dir=text_recognition_dir,
        use_table_recognition=False,
        use_seal_recognition=False,
        use_chart_recognition=False,
        use_formula_recognition=False,
    )
    return model

st.title("PP-Structure V3 Demo")

# Model download section
st.subheader("Model Management")
col1, col2 = st.columns([1, 3])

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

model = load_model()
uploaded_file = st.file_uploader("Upload an image for layout inference", type=["jpg", "jpeg", "png"])

# Check if file has changed
if uploaded_file is not None:
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