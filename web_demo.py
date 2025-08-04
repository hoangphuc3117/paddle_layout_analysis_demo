import streamlit as st

# Set page config for fullscreen (wide) layout
st.set_page_config(page_title="PP-Structure V3 Demo", layout="wide")
from paddleocr import PPStructureV3
import os
import json
import cv2
import numpy as np
import concurrent.futures
import shutil

# Import utility functions
from utils import get_model_paths, upload_image_to_server, image_ocr, sinonom_transliteration, sinonom_prose_translation
from processing import process_layout_ocr_mapping, create_layout_summary

st.title("PP-Structure V3 Demo")

layout_detection_dir, text_detection_dir, text_recognition_dir = get_model_paths()

# Add Overlap Ratio threshold configuration
st.sidebar.header("⚙️ Configuration")
min_overlap_threshold = st.sidebar.slider(
    "Minimum Overlap Ratio Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Minimum overlap ratio threshold for text-to-layout assignment. Higher values require more OCR text coverage within layout."
)

st.sidebar.markdown(f"**Current Overlap Ratio threshold:** {min_overlap_threshold:.2f}")
st.sidebar.markdown("- **0.0-0.3**: Very permissive (allows minimal OCR text coverage)")
st.sidebar.markdown("- **0.3-0.7**: Moderate coverage required")
st.sidebar.markdown("- **0.7+**: High coverage required (most of OCR text must be within layout)")

uploaded_file = st.file_uploader("Upload an image for layout inference", type=["jpg", "jpeg", "png"])

if 'last_file' not in st.session_state:
    st.session_state.last_file = None
    
if uploaded_file is not None:
    if st.session_state.last_file != uploaded_file.name:
        st.session_state.last_file = uploaded_file.name
        # No need to rerun, can process directly
        # st.rerun()

    # Read image into memory
    image_bytes = uploaded_file.getvalue()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Define tasks to run in parallel
    def api_task():
        try:
            # Upload image
            upload_result = upload_image_to_server(image_bytes, uploaded_file.name)
            upload_result_data = json.loads(upload_result.content.decode('utf-8'))
            upload_file_name = upload_result_data['data']['file_name']

            # OCR
            ocr_result = image_ocr(upload_file_name, 1, 1, 1, 1)
            ocr_data = json.loads(ocr_result.content.decode('utf-8'))
            
            # Get OCR text for translation
            ocr_text_list = ocr_data['data']['result_ocr_text']
            ocr_text_string = '\n'.join(ocr_text_list)
            
            # Transliteration
            transliteration_result = sinonom_transliteration(ocr_text_string, 1, 1)
            transliteration_data = json.loads(transliteration_result.content.decode('utf-8'))
            
            # Prose translation
            prose_result = sinonom_prose_translation(ocr_text_string, 1)
            prose_data = json.loads(prose_result.content.decode('utf-8'))
            
            return {
                'ocr_data': ocr_data,
                'transliteration_data': transliteration_data,
                'prose_data': prose_data,
                'status': 'success'
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def model_prediction_task():
        try:
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
            output = model.predict(img_np, batch_size=1, layout_nms=True)
            
            output_dir = "output"
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)

            for res in output:
                res.save_to_img(output_dir)

            # Extract layout detection results
            layout_det_res = output[0]['layout_det_res'].json
            
            return {'layout_det_res': layout_det_res, 'status': 'success'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    # Run both tasks in parallel
    with st.spinner('Processing... Please wait while we analyze your image and call APIs.'):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            api_future = executor.submit(api_task)
            model_future = executor.submit(model_prediction_task)
            
            api_result = api_future.result()
            model_result = model_future.result()

    # Process results if both succeeded
    if api_result['status'] == 'success' and model_result['status'] == 'success':
        # Process mapping with configurable overlap ratio threshold
        mapping_result = process_layout_ocr_mapping(
            model_result['layout_det_res'],
            api_result['ocr_data'],
            api_result['transliteration_data'],
            api_result['prose_data'],
            min_overlap_threshold=min_overlap_threshold
        )
        
        # Create summary
        summary = create_layout_summary(mapping_result)
        
        # Display results
        st.success("Processing completed successfully!")
        
        # Display Overlap Ratio statistics
        st.info(f"🔍 **Overlap Ratio Analysis** (threshold: {min_overlap_threshold:.2f})")
        total_texts = sum(len(data['ocr_results']) for data in mapping_result.values())
        if total_texts > 0:
            all_overlaps = []
            for data in mapping_result.values():
                all_overlaps.extend([item.get('overlap_ratio', 0) for item in data['ocr_results']])
            
            avg_overlap = sum(all_overlaps) / len(all_overlaps) if all_overlaps else 0
            min_overlap = min(all_overlaps) if all_overlaps else 0
            max_overlap = max(all_overlaps) if all_overlaps else 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Texts Assigned", total_texts)
            with col2:
                st.metric("Average Overlap", f"{avg_overlap:.3f}")
            with col3:
                st.metric("Min Overlap", f"{min_overlap:.3f}")
            with col4:
                st.metric("Max Overlap", f"{max_overlap:.3f}")
        else:
            st.warning("⚠️ No texts were assigned to layouts with the current overlap ratio threshold. Consider lowering the threshold.")

        # Display images
        import glob
        
        col1, col2 = st.columns(2)
        
        # Display original image
        with col1:
            st.subheader("Original Image")
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        # Display detection result images
        with col2:
            st.subheader("Layout detection")
            # Find all images containing "det_res" in output folder
            output_images = glob.glob(os.path.join("output", "*layout_det_res*.jpg")) + \
                           glob.glob(os.path.join("output", "*layout_det_res*.png")) + \
                           glob.glob(os.path.join("output", "*layout_det_res*.jpeg"))
            
            if output_images:
                for img_path in sorted(output_images):
                    img_name = os.path.basename(img_path)
                    st.image(img_path, caption=f"Result: {img_name}", use_container_width=True)

        # Display results by layout
        st.subheader("📋 Results by Layout Regions")

        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Original Text**")
            for i, layout_data in enumerate(summary, 1):
                st.subheader(f"{layout_data['label_han_with_english']}")
                # Use markdown with container for auto-height
                with st.container():
                    st.markdown(
                        f'<div style="background-color: #f0f2f6; padding: 12px; border-radius: 6px; border: 1px solid #e6e9ef; min-height: 60px; word-wrap: break-word; white-space: pre-wrap;">{layout_data["original_combined"]}</div>',
                        unsafe_allow_html=True
                    )
        
        with col2:
            st.markdown("**Transliteration**")
            for i, layout_data in enumerate(summary, 1):
                st.subheader(f"{layout_data['label_han_viet']}")
                with st.container():
                    st.markdown(
                        f'<div style="background-color: #f0f2f6; padding: 12px; border-radius: 6px; border: 1px solid #e6e9ef; min-height: 60px; word-wrap: break-word; white-space: pre-wrap;">{layout_data["transcribed_combined"]}</div>',
                        unsafe_allow_html=True
                    )
        
        with col3:
            st.markdown("**Prose Translation**")
            for i, layout_data in enumerate(summary, 1):
                st.subheader(f"{layout_data['label_pure_vietnamese']}")
                with st.container():
                    st.markdown(
                        f'<div style="background-color: #f0f2f6; padding: 12px; border-radius: 6px; border: 1px solid #e6e9ef; min-height: 60px; word-wrap: break-word; white-space: pre-wrap;">{layout_data["prose_combined"]}</div>',
                        unsafe_allow_html=True
                    )
        
        st.divider()  # Add separator between layouts
        
        # Raw data section (collapsible)
        with st.expander("Raw API Results", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("OCR Results")
                st.json(api_result['ocr_data'], expanded=False)
            with col2:
                st.subheader("Layout Detection")
                st.json(model_result['layout_det_res'], expanded=False)
    
    else:
        # Handle errors
        if api_result['status'] == 'error':
            st.error(f"API processing failed: {api_result['error']}")
        if model_result['status'] == 'error':
            st.error(f"Model processing failed: {model_result['error']}")
        
        st.warning("❌ Processing failed. Please try again.")