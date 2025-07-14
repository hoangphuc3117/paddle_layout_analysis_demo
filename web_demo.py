
import streamlit as st

# Set page config for fullscreen (wide) layout
st.set_page_config(page_title="PP-DocLayout_plus-L Model Inference Demo", layout="wide")
from paddlex import create_model
import os
from PIL import Image
import tempfile

st.title("PP-DocLayout_plus-L Model Inference Demo")

# Load model once
@st.cache_resource
def load_model():
    model_name = "PP-DocLayout_plus-L"
    model_dir = "/Users/hoangphuc/Documents/Luan_van/code_test/training_paddle_detection_result/best_model/inference"
    model = create_model(model_name=model_name, model_dir=model_dir, device="cpu")
    return model

model = load_model()

uploaded_file = st.file_uploader("Upload an image for layout inference", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_img_path = tmp_file.name

    # Run prediction
    output = model.predict(tmp_img_path, batch_size=1, layout_nms=True)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    result_img_path = os.path.join(output_dir, "result.jpg")
    result_json_path = os.path.join(output_dir, "result.json")
    for res in output:
        res.save_to_img(result_img_path)
        res.save_to_json(result_json_path)

    # Display in two columns
    col1, col2 = st.columns(2)
    with col1:
        st.image(tmp_img_path, caption="Uploaded Image", use_container_width=True)
    with col2:
        st.image(result_img_path, caption="Detected Layout", use_container_width=True)
        st.download_button("Download JSON Result", open(result_json_path, "rb"), file_name="result.json")

    # Clean up temp file
    os.remove(tmp_img_path)