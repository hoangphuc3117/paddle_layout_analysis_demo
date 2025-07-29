
import streamlit as st
import gc
import os
import psutil
import hashlib

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

# Memory cleanup function
def cleanup_memory():
    """Comprehensive memory cleanup"""
    # Force garbage collection multiple times
    for _ in range(5):
        gc.collect()
    
    # Clear any cached variables
    if 'cached_vars' in locals():
        del cached_vars
    
    # Try to clear PaddlePaddle cache if exists
    try:
        import paddle
        if hasattr(paddle, 'device'):
            if paddle.device.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()
                paddle.device.cuda.synchronize()
    except Exception as e:
        print(f"Paddle cleanup error: {e}")
    
    # Force Python to release memory back to OS
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except:
        pass

def cleanup_session_state():
    """Clean up session state and temporary files"""
    # Clean up temp file if exists
    if 'temp_file_path' in st.session_state and st.session_state.temp_file_path and os.path.exists(st.session_state.temp_file_path):
        try:
            os.unlink(st.session_state.temp_file_path)
        except:
            pass
    
    # Clean up output directory
    output_dir = "output"
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
        except:
            pass
    
    # Clear results from memory first
    if 'results' in st.session_state and st.session_state.results:
        del st.session_state.results
    
    # Reset session state
    st.session_state.current_file_id = None
    st.session_state.results = None
    st.session_state.temp_file_path = None
    st.session_state.memory_stats = {}
    
    # Force cleanup
    cleanup_memory()

def force_model_cleanup():
    """Force cleanup of model resources"""
    try:
        # Clear the cached model
        st.cache_resource.clear()
        
        # Force garbage collection
        cleanup_memory()
        
    except Exception as e:
        print(f"Model cleanup error: {e}")

def get_file_id(uploaded_file):
    """Generate unique ID for uploaded file"""
    if uploaded_file is None:
        return None
    
    # Create hash from file content and metadata
    file_content = uploaded_file.read()
    uploaded_file.seek(0)  # Reset file pointer
    
    file_info = f"{uploaded_file.name}_{uploaded_file.size}_{len(file_content)}"
    return hashlib.md5(file_info.encode()).hexdigest()

def clear_streamlit_cache():
    """Clear Streamlit cache"""
    st.cache_resource.clear()
    st.cache_data.clear()

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # MB

# Load model once
@st.cache_resource
def load_model():
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

with st.sidebar:
    st.header("Memory Management")
    
    # Display current memory usage
    current_memory = get_memory_usage()
    st.metric("Current Memory Usage", f"{current_memory:.1f} MB")
    
    st.divider()
    
    # Clear cache button
    if st.button("� Clear All Cache"):
        initial_memory = get_memory_usage()
        clear_streamlit_cache()
        cleanup_session_state()
        final_memory = get_memory_usage()
        memory_freed = initial_memory - final_memory
        st.success(f"Cache cleared! Freed: {memory_freed:.1f} MB")
        st.rerun()

model = load_model()

uploaded_file = st.file_uploader("Upload an image for layout inference", type=["jpg", "jpeg", "png"])

# Check if file has changed
current_file_id = get_file_id(uploaded_file)

clear_streamlit_cache()

# If no file is uploaded, clean up and show message
if uploaded_file is None:
    # clear_streamlit_cache()
    # cleanup_session_state()
    # st.rerun()
    if st.session_state.current_file_id is not None:
        cleanup_session_state()
    st.info("Please upload an image to start analysis.")
    
# If file has changed, clean up previous results and process new file
elif current_file_id != st.session_state.current_file_id:
    # Clean up previous results
    # clear_streamlit_cache()
    # cleanup_session_state()
    # st.rerun()
    
    # Update current file ID
    st.session_state.current_file_id = current_file_id
    
    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        st.session_state.temp_file_path = tmp_file.name

    # Run prediction with loading spinner
    with st.spinner('Processing... Please wait while we analyze your image.'):
        if not st.session_state.model_loaded:
            st.session_state.model_loaded = True
        

        memory_before_pred = get_memory_usage()
        st.session_state.memory_stats['memory_before'] = memory_before_pred

        print(f"Processing image: {st.session_state.temp_file_path}")
        output = model.predict(st.session_state.temp_file_path, batch_size=1)
        
        memory_after_pred = get_memory_usage()
        memory_used = memory_after_pred - memory_before_pred
        
        st.session_state.memory_stats['memory_after'] = memory_after_pred
        st.session_state.memory_stats['memory_used'] = memory_used

        output_dir = "output"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.makedirs(output_dir, exist_ok=True)

        # Save results
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

    clear_streamlit_cache()
    model = load_model()
