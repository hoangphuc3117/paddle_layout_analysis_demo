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
        PIL Image với bounding boxes được vẽ hoặc None nếu có lỗi
    """
    try:
        # Validate input data
        if image is None:
            st.error("Lỗi: Ảnh đầu vào không hợp lệ")
            return None
            
        if not ocr_data or 'data' not in ocr_data or 'result_bbox' not in ocr_data['data']:
            st.warning("Không có dữ liệu OCR để hiển thị bounding boxes")
            return image if isinstance(image, Image.Image) else Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            if image.size == 0:
                st.error("Lỗi: Ảnh đầu vào rỗng")
                return None
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image = image.copy()
        
        draw = ImageDraw.Draw(image)
    except Exception as e:
        st.error(f"Lỗi khi xử lý ảnh đầu vào")
        return None
    
    try:
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
                
                if 'ocr_results' in data:
                    for ocr_item in data['ocr_results']:
                        if 'original_text' in ocr_item:
                            text_to_layout[ocr_item['original_text']] = layout_label
        
        # Vẽ bounding boxes
        bbox_list = ocr_data['data']['result_bbox']
        if not bbox_list:
            st.warning("Không tìm thấy text boxes trong kết quả OCR")
            return image
            
        for bbox_info in bbox_list:
            try:
                # Validate bbox_info structure
                if not bbox_info or len(bbox_info) < 2:
                    continue
                    
                bbox_coords = bbox_info[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text_info = bbox_info[1]
                
                if not text_info or len(text_info) < 2:
                    continue
                    
                text = text_info[0]
                confidence = text_info[1]
                
                # Validate bbox coordinates
                if not bbox_coords or len(bbox_coords) < 4:
                    continue
                
                # Convert 4-point bbox to rectangle coordinates
                x_coords = [point[0] for point in bbox_coords if len(point) >= 2]
                y_coords = [point[1] for point in bbox_coords if len(point) >= 2]
                
                if len(x_coords) < 4 or len(y_coords) < 4:
                    continue
                    
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
            except Exception as e:
                # Log individual bbox error but continue processing
                st.warning(f"Lỗi khi vẽ một bounding box")
                continue
        
        return image
        
    except Exception as e:
        st.error(f"Lỗi khi vẽ OCR bounding boxes")
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

# Validate model paths at startup
try:
    layout_detection_dir, text_detection_dir, text_recognition_dir = get_model_paths()
    
    if not layout_detection_dir or not os.path.exists(layout_detection_dir):
        st.error("❌ Không tìm thấy thư mục model layout detection. Vui lòng kiểm tra cấu hình.")
        st.stop()
        
except Exception as e:
    st.error(f"❌ Lỗi khi tải model paths: {str(e)}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload ảnh để phân tích layout", 
    type=["jpg", "jpeg", "png"],
    help="Chọn file ảnh định dạng JPG, JPEG hoặc PNG. Kích thước tối đa: 50MB"
)

if 'last_file' not in st.session_state:
    st.session_state.last_file = None
    
if uploaded_file is not None:
    # Validate uploaded file
    try:
        # Check file size (limit to 10MB)
        file_size = len(uploaded_file.getvalue())
        if file_size > 10 * 1024 * 1024:  # 10MB
            st.error("❌ Kích thước file quá lớn! Vui lòng chọn file nhỏ hơn 10MB.")
            st.stop()
        
        # Check file type
        if uploaded_file.type not in ['image/jpeg', 'image/jpg', 'image/png']:
            st.error("❌ Định dạng file không được hỗ trợ! Vui lòng chọn file JPG, JPEG hoặc PNG.")
            st.stop()
            
        if st.session_state.last_file != uploaded_file.name:
            st.session_state.last_file = uploaded_file.name

        # Read and validate image
        image_bytes = uploaded_file.getvalue()
        if len(image_bytes) == 0:
            st.error("❌ File ảnh rỗng hoặc không hợp lệ!")
            st.stop()
            
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_np is None or img_np.size == 0:
            st.error("❌ Không thể đọc ảnh! Vui lòng kiểm tra lại file ảnh.")
            st.stop()
            
        # Check image dimensions
        height, width = img_np.shape[:2]
        if height < 50 or width < 50:
            st.error("❌ Ảnh quá nhỏ! Kích thước tối thiểu là 50x50 pixels.")
            st.stop()
            
        if height > 5000 or width > 5000:
            st.warning("⚠️ Ảnh có kích thước lớn, quá trình xử lý có thể mất nhiều thời gian.")

    except Exception as e:
        st.error(f"❌ Lỗi khi xử lý file ảnh")
        st.stop()

    # Define tasks to run in parallel
    def api_task():
        try:
            # Upload image
            upload_result = upload_image_to_server(image_bytes, uploaded_file.name)
            
            if upload_result is None:
                return {'status': 'error', 'error': 'Không thể kết nối đến server upload'}
            
            if upload_result.status_code != 200:
                return {'status': 'error', 'error': f'Lỗi upload ảnh: HTTP {upload_result.status_code}'}
            
            try:
                upload_result_data = json.loads(upload_result.content.decode('utf-8'))
            except json.JSONDecodeError:
                return {'status': 'error', 'error': 'Server upload trả về dữ liệu không hợp lệ'}
            
            if 'data' not in upload_result_data or 'file_name' not in upload_result_data['data']:
                return {'status': 'error', 'error': 'Không nhận được tên file từ server upload'}
                
            upload_file_name = upload_result_data['data']['file_name']

            # OCR
            ocr_result = image_ocr(upload_file_name, 1, 1, 1, 1)
            
            if ocr_result is None:
                return {'status': 'error', 'error': 'Không thể kết nối đến API OCR'}
            
            if ocr_result.status_code != 200:
                return {'status': 'error', 'error': f'Lỗi API OCR: HTTP {ocr_result.status_code}'}
            
            try:
                ocr_data = json.loads(ocr_result.content.decode('utf-8'))
            except json.JSONDecodeError:
                return {'status': 'error', 'error': 'API OCR trả về dữ liệu không hợp lệ'}
            
            # Validate OCR data structure
            if 'data' not in ocr_data:
                return {'status': 'error', 'error': 'Dữ liệu OCR thiếu thông tin cần thiết'}
                
            if 'result_ocr_text' not in ocr_data['data'] or 'result_bbox' not in ocr_data['data']:
                return {'status': 'error', 'error': 'Không có kết quả OCR trong response'}
            
            # Check if OCR found any text
            ocr_text_list = ocr_data['data']['result_ocr_text']
            if not ocr_text_list or len(ocr_text_list) == 0:
                return {
                    'ocr_data': ocr_data,
                    'transliteration_data': {'data': {'result_transliteration': []}},
                    'prose_data': {'data': {'result_prose_translation': []}},
                    'status': 'success_no_text',
                    'message': 'Không tìm thấy text trong ảnh'
                }
            
            ocr_text_string = '\n'.join(ocr_text_list)
            
            # Transliteration
            transliteration_result = sinonom_transliteration(ocr_text_string, 1, 1)
            
            if transliteration_result is None:
                return {'status': 'error', 'error': 'Không thể kết nối đến API dịch âm'}
            
            if transliteration_result.status_code != 200:
                return {'status': 'error', 'error': f'Lỗi API dịch âm: HTTP {transliteration_result.status_code}'}
            
            try:
                transliteration_data = json.loads(transliteration_result.content.decode('utf-8'))
            except json.JSONDecodeError:
                return {'status': 'error', 'error': 'API dịch âm trả về dữ liệu không hợp lệ'}
            
            # Prose translation
            prose_result = sinonom_prose_translation(ocr_text_string, 1)
            
            if prose_result is None:
                return {'status': 'error', 'error': 'Không thể kết nối đến API dịch nghĩa'}
            
            if prose_result.status_code != 200:
                return {'status': 'error', 'error': f'Lỗi API dịch nghĩa: HTTP {prose_result.status_code}'}
            
            try:
                prose_data = json.loads(prose_result.content.decode('utf-8'))
            except json.JSONDecodeError:
                return {'status': 'error', 'error': 'API dịch nghĩa trả về dữ liệu không hợp lệ'}
            
            return {
                'ocr_data': ocr_data,
                'transliteration_data': transliteration_data,
                'prose_data': prose_data,
                'status': 'success'
            }
            
        except ConnectionError:
            return {'status': 'error', 'error': 'Lỗi kết nối mạng. Vui lòng kiểm tra kết nối internet.'}
        except Exception as e:
            return {'status': 'error', 'error': f'Lỗi không xác định trong quá trình gọi API'}
    
    def model_prediction_task():
        try:
            # Validate model paths
            if not layout_detection_dir or not os.path.exists(layout_detection_dir):
                return {'status': 'error', 'error': 'Không tìm thấy thư mục model layout detection'}
            
            model_name = "PP-DocLayout-L"
            
            # Create model with error handling
            try:
                model = create_model(model_name=model_name, model_dir=layout_detection_dir)
            except Exception as e:
                return {'status': 'error', 'error': f'Không thể tạo model {model_name}'}
            
            if model is None:
                return {'status': 'error', 'error': f'Model {model_name} không được khởi tạo thành công'}
            
            # Run prediction
            try:
                output = model.predict(img_np, batch_size=1, layout_nms=True)
            except Exception as e:
                return {'status': 'error', 'error': f'Lỗi khi chạy model prediction'}
            
            if not output:
                return {'status': 'error', 'error': 'Model không trả về kết quả'}
            
            # Create output directory
            output_dir = "output"
            try:
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
                os.makedirs(output_dir, exist_ok=True)
            except Exception as e:
                return {'status': 'error', 'error': f'Không thể tạo thư mục output'}

            # Save results
            layout_det_res = None
            try:
                for res in output:
                    res.save_to_img(output_dir)
                    layout_det_res = res.json
            except Exception as e:
                return {'status': 'error', 'error': f'Lỗi khi lưu kết quả'}
            
            if layout_det_res is None:
                return {'status': 'error', 'error': 'Không có kết quả layout detection'}
            
            # Validate layout detection results
            if not isinstance(layout_det_res, list) or len(layout_det_res) == 0:
                return {
                    'layout_det_res': [],
                    'status': 'success_no_layout',
                    'message': 'Không tìm thấy layout nào trong ảnh'
                }
        
            return {'layout_det_res': layout_det_res, 'status': 'success'}
            
        except MemoryError:
            return {'status': 'error', 'error': 'Hết bộ nhớ khi xử lý ảnh. Vui lòng thử với ảnh có kích thước nhỏ hơn.'}
        except Exception as e:
            return {'status': 'error', 'error': f'Lỗi không xác định trong model prediction'}

    # Run both tasks in parallel
    with st.spinner('Đang xử lý... Vui lòng đợi trong khi chúng tôi phân tích ảnh và gọi API.'):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                api_future = executor.submit(api_task)
                model_future = executor.submit(model_prediction_task)
                
                # Wait for results with timeout
                try:
                    api_result = api_future.result(timeout=120)  # 2 minutes timeout
                    model_result = model_future.result(timeout=120)  # 2 minutes timeout
                except concurrent.futures.TimeoutError:
                    st.error("❌ Quá thời gian xử lý! Vui lòng thử lại với ảnh nhỏ hơn hoặc kiểm tra kết nối mạng.")
                    st.stop()
        except Exception as e:
            st.error(f"❌ Lỗi trong quá trình xử lý song song")
            st.stop()

    # Check results and handle different scenarios
    api_success = api_result['status'] in ['success', 'success_no_text']
    model_success = model_result['status'] in ['success', 'success_no_layout']
    
    if not api_success and not model_success:
        st.error("❌ Cả API và Model đều gặp lỗi:")
        st.error(f"• API: {api_result['error']}")
        st.error(f"• Model: {model_result['error']}")
        st.warning("Vui lòng thử lại hoặc kiểm tra kết nối mạng.")
        st.stop()
    elif not api_success:
        st.error(f"❌ Lỗi API: {api_result['error']}")
        st.warning("Không thể xử lý OCR và dịch thuật. Chỉ hiển thị kết quả layout detection.")
        # Continue with only model results
    elif not model_success:
        st.error(f"❌ Lỗi Model: {model_result['error']}")
        st.warning("Không thể phân tích layout. Chỉ hiển thị kết quả OCR.")
        # Continue with only API results

    # Handle case where OCR found no text
    if api_result['status'] == 'success_no_text':
        st.warning("⚠️ " + api_result['message'])
        st.info("Gợi ý: Thử với ảnh có chất lượng tốt hơn hoặc có chứa text rõ ràng hơn.")
    
    # Handle case where layout detection found no layouts
    if model_result['status'] == 'success_no_layout':
        st.warning("⚠️ " + model_result['message'])
        st.info("Gợi ý: Thử với ảnh có layout rõ ràng hơn hoặc có cấu trúc văn bản.")

    # Process results if at least one succeeded
    if api_success and model_success and api_result['status'] == 'success' and model_result['status'] == 'success':
        try:
            # Process mapping with configurable overlap ratio threshold
            mapping_result = process_layout_ocr_mapping(
                model_result['layout_det_res'],
                api_result['ocr_data'],
                api_result['transliteration_data'],
                api_result['prose_data'],
                min_overlap_threshold=min_overlap_threshold
            )
            
            if not mapping_result:
                st.warning("⚠️ Không thể tạo mapping giữa layout và OCR")
                mapping_result = {}
            
            # Create summary
            summary = create_layout_summary(mapping_result)
            
        except Exception as e:
            st.error(f"❌ Lỗi khi xử lý mapping")
            st.warning("Hiển thị kết quả riêng lẻ...")
            mapping_result = {}
            summary = []
        
        # Display results
        st.success("✅ Xử lý thành công!")
        
        # Create OCR visualization with error handling
        ocr_image = None
        if api_success and 'ocr_data' in api_result:
            try:
                ocr_image = draw_ocr_bboxes(
                    img_np, 
                    api_result['ocr_data'], 
                    mapping_result,
                    show_text_labels=True,
                    show_confidence=False
                )
            except Exception as e:
                st.warning(f"⚠️ Không thể tạo OCR visualization: {str(e)}")

        # Display images with error handling
        try:
            import glob
            
            col1, col2, col3 = st.columns(3)
            
            # Display original image
            with col1:
                st.subheader("Ảnh gốc")
                try:
                    st.image(uploaded_file, caption="Ảnh đã upload", use_container_width=True)
                except Exception as e:
                    st.error(f"Không thể hiển thị ảnh gốc")
            
            # Display OCR bounding boxes
            with col2:
                st.subheader("OCR Bounding Boxes")
                if ocr_image is not None:
                    try:
                        st.image(ocr_image, caption="OCR Detection với Layout Assignment", use_container_width=True)
                    except Exception as e:
                        st.error(f"Không thể hiển thị OCR boxes")
                else:
                    st.warning("Không có dữ liệu OCR để hiển thị")
            
            # Display detection result images
            with col3:
                st.subheader("Layout Detection")
                try:
                    # Find all images containing "det_res" in output folder
                    output_images = glob.glob(os.path.join("output", "*res*.jpg")) + \
                                   glob.glob(os.path.join("output", "*res*.png")) + \
                                   glob.glob(os.path.join("output", "*res*.jpeg"))
                    
                    if output_images:
                        for img_path in sorted(output_images):
                            try:
                                img_name = os.path.basename(img_path)
                                st.image(img_path, caption=f"Kết quả: {img_name}", use_container_width=True)
                            except Exception as e:
                                st.warning(f"Không thể hiển thị {img_path}: {str(e)}")
                    else:
                        st.warning("Không tìm thấy ảnh kết quả layout detection")
                except Exception as e:
                    st.error(f"Lỗi khi tìm ảnh kết quả")
        except Exception as e:
            st.error(f"Lỗi khi hiển thị ảnh")

        # Display results by layout with error handling
        if summary and len(summary) > 0:
            try:
                st.subheader("📋 Kết quả")

                # Display each layout as a separate section with 3 columns
                for i, layout_data in enumerate(summary, 1):
                    try:
                        # Create 3 columns for this layout
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.subheader(f"{layout_data.get('label_han_with_english', 'N/A')}")
                            # Create list display for original texts
                            original_list_html = "<div style='background-color: #f0f2f6; padding: 12px; border-radius: 6px; border: 1px solid #e6e9ef; min-height: 60px;'>"
                            if layout_data.get('original_texts'):
                                for idx, text in enumerate(layout_data['original_texts'], 1):
                                    original_list_html += f"<div style='margin-bottom: 8px; padding: 4px 0; border-bottom: 1px solid #ddd;'>{text}</div>"
                            else:
                                original_list_html += f"<div>{layout_data.get('original_combined', 'Không có dữ liệu')}</div>"
                            original_list_html += "</div>"
                            st.markdown(original_list_html, unsafe_allow_html=True)
                        
                        with col2:
                            st.subheader(f"{layout_data.get('label_han_viet', 'N/A')}")
                            # Create list display for transcribed texts
                            transcribed_list_html = "<div style='background-color: #f0f2f6; padding: 12px; border-radius: 6px; border: 1px solid #e6e9ef; min-height: 60px;'>"
                            if layout_data.get('transcribed_texts'):
                                for idx, text in enumerate(layout_data['transcribed_texts'], 1):
                                    transcribed_list_html += f"<div style='margin-bottom: 8px; padding: 4px 0; border-bottom: 1px solid #ddd;'>{text}</div>"
                            else:
                                transcribed_list_html += f"<div>{layout_data.get('transcribed_combined', 'Không có dữ liệu')}</div>"
                            transcribed_list_html += "</div>"
                            st.markdown(transcribed_list_html, unsafe_allow_html=True)
                        
                        with col3:
                            st.subheader(f"{layout_data.get('label_pure_vietnamese', 'N/A')}")
                            # Create list display for prose texts
                            prose_list_html = "<div style='background-color: #f0f2f6; padding: 12px; border-radius: 6px; border: 1px solid #e6e9ef; min-height: 60px;'>"
                            if layout_data.get('prose_texts'):
                                for idx, text in enumerate(layout_data['prose_texts'], 1):
                                    prose_list_html += f"<div style='margin-bottom: 8px; padding: 4px 0; border-bottom: 1px solid #ddd;'>{text}</div>"
                            else:
                                prose_list_html += f"<div>{layout_data.get('prose_combined', 'Không có dữ liệu')}</div>"
                            prose_list_html += "</div>"
                            st.markdown(prose_list_html, unsafe_allow_html=True)
                        
                        # Add separator between layouts except for the last one
                        if i < len(summary):
                            st.divider()
                    except Exception as e:
                        st.error(f"Lỗi khi hiển thị layout {i}")
                        continue
                        
                st.divider()  # Add separator between layouts
            except Exception as e:
                st.error(f"Lỗi khi hiển thị kết quả")
        else:
            st.warning("⚠️ Không có kết quả để hiển thị")
        
        # Raw data section (collapsible) with error handling
        try:
            with st.expander("Dữ liệu thô từ API", expanded=False):
                # First row - OCR and Layout Detection
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("OCR")
                    if api_success and 'ocr_data' in api_result:
                        try:
                            st.json(api_result['ocr_data'], expanded=False)
                        except Exception as e:
                            st.error(f"Lỗi hiển thị dữ liệu OCR")
                    else:
                        st.warning("Không có dữ liệu OCR")
                        
                with col2:
                    st.subheader("Layout Detection")
                    if model_success and 'layout_det_res' in model_result:
                        try:
                            st.json(model_result['layout_det_res'], expanded=False)
                        except Exception as e:
                            st.error(f"Lỗi hiển thị dữ liệu Layout")
                    else:
                        st.warning("Không có dữ liệu Layout Detection")
                
                # Second row - Transliteration and Prose Translation
                col3, col4 = st.columns(2)
                with col3:
                    st.subheader("Dịch âm")
                    if api_success and 'transliteration_data' in api_result:
                        try:
                            st.json(api_result['transliteration_data'], expanded=False)
                        except Exception as e:
                            st.error(f"Lỗi hiển thị dữ liệu dịch âm")
                    else:
                        st.warning("Không có dữ liệu dịch âm")
                        
                with col4:
                    st.subheader("Dịch nghĩa")
                    if api_success and 'prose_data' in api_result:
                        try:
                            st.json(api_result['prose_data'], expanded=False)
                        except Exception as e:
                            st.error(f"Lỗi hiển thị dữ liệu dịch nghĩa")
                    else:
                        st.warning("Không có dữ liệu dịch nghĩa")
        except Exception as e:
            st.error(f"Lỗi khi hiển thị dữ liệu thô")
    
    else:
        # Handle partial success or complete failure
        if not api_success and api_result['status'] == 'error':
            st.error(f"❌ Lỗi API: {api_result['error']}")
        if not model_success and model_result['status'] == 'error':
            st.error(f"❌ Lỗi Model: {model_result['error']}")
        
        # Show what we can show
        if model_success and model_result['status'] == 'success':
            st.info("📋 Chỉ hiển thị kết quả Layout Detection:")
            try:
                import glob
                output_images = glob.glob(os.path.join("output", "*res*.jpg")) + \
                               glob.glob(os.path.join("output", "*res*.png")) + \
                               glob.glob(os.path.join("output", "*res*.jpeg"))
                
                if output_images:
                    for img_path in sorted(output_images):
                        img_name = os.path.basename(img_path)
                        st.image(img_path, caption=f"Layout Detection: {img_name}", use_container_width=True)
            except Exception as e:
                st.error(f"Không thể hiển thị kết quả layout")
                
        if api_success and api_result['status'] == 'success':
            st.info("📋 Chỉ hiển thị kết quả OCR:")
            try:
                ocr_texts = api_result['ocr_data']['data']['result_ocr_text']
                for i, text in enumerate(ocr_texts, 1):
                    st.write(f"{i}. {text}")
            except Exception as e:
                st.error(f"Không thể hiển thị OCR text")
        
        st.warning("⚠️ Một số chức năng gặp lỗi. Vui lòng thử lại hoặc kiểm tra kết nối mạng.")