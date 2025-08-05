import streamlit as st

# Set page config for fullscreen (wide) layout
st.set_page_config(page_title="PP-Structure V3 Demo", layout="wide")
from paddlex import create_model
import os
import json
import cv2
import numpy as np
import concurrent.futures
import shutil
from PIL import Image, ImageDraw, ImageFont
import random

# Import utility functions
from utils import get_model_paths, upload_image_to_server, image_ocr, sinonom_transliteration, sinonom_prose_translation
from processing import process_layout_ocr_mapping, create_layout_summary

def draw_ocr_bboxes(image, ocr_data, mapping_result=None, show_text_labels=False, show_confidence=False):
    """
    Vẽ bounding boxes của OCR lên hình ảnh
    
    Args:
        image: PIL Image hoặc numpy array
        ocr_data: Dữ liệu OCR từ API
        mapping_result: Kết quả mapping (optional) để phân màu theo layout
        show_text_labels: Hiển thị text labels
        show_confidence: Hiển thị confidence scores
    
    Returns:
        PIL Image với bounding boxes được vẽ
    """
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        image = image.copy()
    
    draw = ImageDraw.Draw(image)
    
    # Tạo mapping từ text đến layout label nếu có mapping_result
    text_to_layout = {}
    layout_colors = {}
    if mapping_result:
        color_palette = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57',
            '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43'
        ]
        
        for i, (layout_label, data) in enumerate(mapping_result.items()):
            color = color_palette[i % len(color_palette)]
            layout_colors[layout_label] = color
            
            for ocr_item in data['ocr_results']:
                text_to_layout[ocr_item['original_text']] = layout_label
    
    # Vẽ bounding boxes
    for bbox_info in ocr_data['data']['result_bbox']:
        bbox_coords = bbox_info[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        text = bbox_info[1][0]
        confidence = bbox_info[1][1]
        
        # Convert 4-point bbox to rectangle coordinates
        x_coords = [point[0] for point in bbox_coords]
        y_coords = [point[1] for point in bbox_coords]
        x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
        
        # Chọn màu dựa trên layout mapping
        if text in text_to_layout:
            layout_label = text_to_layout[text]
            color = layout_colors[layout_label]
            thickness = 3
        else:
            color = '#FF0000'  # Đỏ cho text không được assign
            thickness = 2
        
        # Vẽ bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        
        # Vẽ text label và confidence (tùy chọn)
        if show_text_labels or show_confidence:
            try:
                # Cố gắng load font, nếu không được thì dùng default
                font = ImageFont.truetype("arial.ttf", 10)
            except:
                font = ImageFont.load_default()
            
            # Tạo label text
            label_parts = []
            if show_text_labels and len(text) < 20:  # Chỉ hiển thị text ngắn
                label_parts.append(text)
            if show_confidence:
                label_parts.append(f"({confidence:.2f})")
            
            if label_parts:
                label_text = " ".join(label_parts)
                
                # Vẽ background cho text
                text_bbox = draw.textbbox((x1, max(0, y1-15)), label_text, font=font)
                draw.rectangle(text_bbox, fill=color, outline=color)
                draw.text((x1, max(0, y1-15)), label_text, fill='white', font=font)
    
    return image

def create_legend(mapping_result):
    """
    Tạo legend cho các màu layout
    """
    if not mapping_result:
        return ""
    
    color_palette = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57',
        '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43'
    ]
    
    legend_html = "<div style='padding: 10px; background-color: #f0f2f6; border-radius: 6px;'>"
    legend_html += "<h4>Layout Color Legend:</h4>"
    
    for i, (layout_label, data) in enumerate(mapping_result.items()):
        color = color_palette[i % len(color_palette)]
        text_count = len(data['ocr_results'])
        legend_html += f"<div style='margin: 5px 0;'>"
        legend_html += f"<span style='display: inline-block; width: 20px; height: 15px; background-color: {color}; border: 1px solid #ccc; margin-right: 8px;'></span>"
        legend_html += f"<strong>{layout_label}</strong> ({text_count} texts)"
        legend_html += f"</div>"
    
    legend_html += "<div style='margin: 5px 0;'>"
    legend_html += f"<span style='display: inline-block; width: 20px; height: 15px; background-color: #FF0000; border: 1px solid #ccc; margin-right: 8px;'></span>"
    legend_html += f"<strong>Unassigned texts</strong>"
    legend_html += f"</div>"
    
    legend_html += "</div>"
    return legend_html

st.title("Layout Detection Demo (Thông thường, Hán, Dọc, In)")

# Add Overlap Ratio threshold configuration
min_overlap_threshold = 0.5

layout_detection_dir, text_detection_dir, text_recognition_dir = get_model_paths()

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
        # try:
        model_name = "PP-DocLayout-L"
        model = create_model(model_name=model_name, model_dir=layout_detection_dir)
        output = model.predict(img_np, batch_size=1, layout_nms=True)
        
        output_dir = "output"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        for res in output:
            res.save_to_img(output_dir)
            layout_det_res = res.json
        
        return {'layout_det_res': layout_det_res, 'status': 'success'}
        # except Exception as e:
        #     return {'status': 'error', 'error': str(e)}

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
        
        # Create OCR visualization
        ocr_image = draw_ocr_bboxes(
            img_np, 
            api_result['ocr_data'], 
            mapping_result,
            show_text_labels=True,
            show_confidence=False
        )

        # Display images
        import glob
        
        col1, col2, col3 = st.columns(3)
        
        # Display original image
        with col1:
            st.subheader("Original Image")
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        # Display OCR bounding boxes
        with col2:
            st.subheader("OCR Bounding Boxes")
            st.image(ocr_image, caption="OCR Detection with Layout Assignment", use_container_width=True)
        
        # Display detection result images
        with col3:
            st.subheader("Layout Detection")
            # Find all images containing "det_res" in output folder
            output_images = glob.glob(os.path.join("output", "*res*.jpg")) + \
                           glob.glob(os.path.join("output", "*res*.png")) + \
                           glob.glob(os.path.join("output", "*res*.jpeg"))
            
            if output_images:
                for img_path in sorted(output_images):
                    img_name = os.path.basename(img_path)
                    st.image(img_path, caption=f"Result: {img_name}", use_container_width=True)

        # Display results by layout
        st.subheader("📋 Kết quả")

        # Display each layout as a separate section with 3 columns
        for i, layout_data in enumerate(summary, 1):
            # Create 3 columns for this layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader(f"{layout_data['label_han_with_english']}")
                # Create list display for original texts
                original_list_html = "<div style='background-color: #f0f2f6; padding: 12px; border-radius: 6px; border: 1px solid #e6e9ef; min-height: 60px;'>"
                if layout_data.get('original_texts'):
                    for idx, text in enumerate(layout_data['original_texts'], 1):
                        original_list_html += f"<div style='margin-bottom: 8px; padding: 4px 0; border-bottom: 1px solid #ddd;'>{text}</div>"
                else:
                    original_list_html += f"<div>{layout_data['original_combined']}</div>"
                original_list_html += "</div>"
                st.markdown(original_list_html, unsafe_allow_html=True)
            
            with col2:
                st.subheader(f"{layout_data['label_han_viet']}")
                # Create list display for transcribed texts
                transcribed_list_html = "<div style='background-color: #f0f2f6; padding: 12px; border-radius: 6px; border: 1px solid #e6e9ef; min-height: 60px;'>"
                if layout_data.get('transcribed_texts'):
                    for idx, text in enumerate(layout_data['transcribed_texts'], 1):
                        transcribed_list_html += f"<div style='margin-bottom: 8px; padding: 4px 0; border-bottom: 1px solid #ddd;'>{text}</div>"
                else:
                    transcribed_list_html += f"<div>{layout_data['transcribed_combined']}</div>"
                transcribed_list_html += "</div>"
                st.markdown(transcribed_list_html, unsafe_allow_html=True)
            
            with col3:
                st.subheader(f"{layout_data['label_pure_vietnamese']}")
                # Create list display for prose texts
                prose_list_html = "<div style='background-color: #f0f2f6; padding: 12px; border-radius: 6px; border: 1px solid #e6e9ef; min-height: 60px;'>"
                if layout_data.get('prose_texts'):
                    for idx, text in enumerate(layout_data['prose_texts'], 1):
                        prose_list_html += f"<div style='margin-bottom: 8px; padding: 4px 0; border-bottom: 1px solid #ddd;'>{text}</div>"
                else:
                    prose_list_html += f"<div>{layout_data['prose_combined']}</div>"
                prose_list_html += "</div>"
                st.markdown(prose_list_html, unsafe_allow_html=True)
            
            # Add separator between layouts except for the last one
            if i < len(summary):
                st.divider()
        
        st.divider()  # Add separator between layouts
        
        # Raw data section (collapsible)
        with st.expander("Raw API Results", expanded=False):
            # First row - OCR and Layout Detection
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("OCR Results")
                st.json(api_result['ocr_data'], expanded=False)
            with col2:
                st.subheader("Layout Detection")
                st.json(model_result['layout_det_res'], expanded=False)
            
            # Second row - Transliteration and Prose Translation
            col3, col4 = st.columns(2)
            with col3:
                st.subheader("Transliteration Results")
                st.json(api_result['transliteration_data'], expanded=False)
            with col4:
                st.subheader("Prose Translation Results")
                st.json(api_result['prose_data'], expanded=False)
    
    else:
        # Handle errors
        if api_result['status'] == 'error':
            st.error(f"API processing failed: {api_result['error']}")
        if model_result['status'] == 'error':
            st.error(f"Model processing failed: {model_result['error']}")
        
        st.warning("❌ Processing failed. Please try again.")