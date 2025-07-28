
import streamlit as st

# Set page config for fullscreen (wide) layout
st.set_page_config(page_title="PP-Structure V3 Demo", layout="wide")
# from paddlex import create_model
from paddleocr import PPStructureV3
import os
import tempfile
import shutil
import glob
import json

st.title("PP-Structure V3 Demo")

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

model = load_model()

uploaded_file = st.file_uploader("Upload an image for layout inference", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_img_path = tmp_file.name

    # Run prediction with loading spinner
    with st.spinner('Processing... Please wait while we analyze your image.'):
        output = model.predict(tmp_img_path, batch_size=1, layout_nms=True)

    output_dir = "output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Save results
    # result_img_path = os.path.join(output_dir, "result.jpg")
    # result_json_path = os.path.join(output_dir, "result.json")
    for res in output:
        res.save_to_img(output_dir)
        res.save_to_json(output_dir)


    st.subheader("Original Image")
    st.image(tmp_img_path, caption="Uploaded Image", use_container_width=False)
    
    st.subheader("Detection Results")
    
    # Get all image files from output directory
    image_files = glob.glob(os.path.join(output_dir, "*.jpg")) + \
                  glob.glob(os.path.join(output_dir, "*.jpeg")) + \
                  glob.glob(os.path.join(output_dir, "*.png"))
    
    # Get all JSON files from output directory
    json_files = glob.glob(os.path.join(output_dir, "*.json"))
    
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

    # Clean up temp file
    os.remove(tmp_img_path)