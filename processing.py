import requests
from utils import BASE_API_URL

# Dictionary for layout label translations
LAYOUT_LABEL_TRANSLATIONS = {
    'compiler': {
        'han_text': '編者 (Compiler)',
        'han_viet': 'Biên giả', 
        'pure_vietnamese': 'Người biên tập'
    },
    'author': {
        'han_text': '作者 (Author)',
        'han_viet': 'Tác giả',
        'pure_vietnamese': 'Tác giả'
    },
    'text': {
        'han_text': '正文 (Text)',
        'han_viet': 'Chính văn',
        'pure_vietnamese': 'Nội dung chính'
    },
    'section title': {
        'han_text': '節名 (Section title)',
        'han_viet': 'Điều mục',
        'pure_vietnamese': 'Tên đoạn'
    },
    'chapter title': {
        'han_text': '編者 (Chapter title)',
        'han_viet': 'Quyển thủ',
        'pure_vietnamese': 'Tên chương'
    },
    'subtitle': {
        'han_text': '副題 (Subtitle)',
        'han_viet': 'Đề phụ',
        'pure_vietnamese': 'Tựa nhỏ'
    },
    'centerfold strip': {
        'han_text': '魂骨條 (Centerfold strip)',
        'han_viet': 'Hồn cốt điều',
        'pure_vietnamese': 'Gáy sách'
    },
    'EOV':{
        'han_text': '結尾 (End of Volume)',
        'han_viet': 'kết vĩ',
        'pure_vietnamese': 'Kết thúc quyển'
    },
    'bibliography':{
        'han_text': '目录 (Bibliography)',
        'han_viet': 'Mục lục',
        'pure_vietnamese': 'Mục lục sách'
    },
    'book number':{
        'han_text': '册号 (Book number)',
        'han_viet': 'Sách hiệu',
        'pure_vietnamese': 'Mã số sách'
    },
    'caption': {
        'han_text': '图注 (Caption)',
        'han_viet': 'đồ chú',
        'pure_vietnamese': 'Ghi chú hình ảnh'
    },
    'colophon': {
        'han_text': '牌记 (Colophon)',
        'han_viet': 'Bài ký',
        'pure_vietnamese': 'Ký hiệu đặc biệt'
    },
    'collation table': {
        'han_text': '校勘表 (Collation table)',
        'han_viet': 'Hiệu khám biểu',
        'pure_vietnamese': 'Bảng hiệu đính'
    },
    'endnote': {
        'han_text': '尾注 (Endnote)',
        'han_viet': 'Vỹ chú',
        'pure_vietnamese': 'Chú thích cuối'
    },
    'ear note': {
        'han_text': '书耳 (Ear note)',
        'han_viet': 'Thư nhĩ',
        'pure_vietnamese': 'Ghi chú tai sách'
    },
    'engraver': {
        'han_text': '刻工名字/刊刻者 (Engraver)',
        'han_viet': 'Khắc công danh tự/San khắc giả',
        'pure_vietnamese': 'Người khắc'
    },
    'figure': {
        'han_text': '插图 (Figure)',
        'han_viet': 'Sáp đồ',
        'pure_vietnamese': 'Hình ảnh'
    },
    'header': {
        'han_text': '书眉 (Header)',
        'han_viet': 'Thư mi',
        'pure_vietnamese': 'Tiêu đề đầu trang'
    },
    'interliner note': {
        'han_text': '夹注 (Interliner note)',
        'han_viet': 'Giáp chú',
        'pure_vietnamese': 'Ghi chú giữa dòng'
    },
    'marginal annotation': {
        'han_text': '邊註 (Marginal annotation)',
        'han_viet': 'Chú thích lề',
        'pure_vietnamese': 'Chú thích lề'
    },
    'sub section title': {
        'han_text': '小节标题 (Sub section title)',
        'han_viet': 'Tiểu tiết đề mục',
        'pure_vietnamese': 'Tiêu đề tiểu đoạn'
    },
    'sutra number': {
        'han_text': '经号 (Sutra number)',
        'han_viet': 'Kinh hiệu',
        'pure_vietnamese': 'Mã số kinh văn'
    },
    'volume number': {
        'han_text': '卷號 (Volume number)',
        'han_viet': 'Quyển thứ',
        'pure_vietnamese': 'Số thứ tự quyển'
    },
    'foliation': {
        'han_text': '页码 (Foliation)',
        'han_viet': 'Diệp mã',
        'pure_vietnamese': 'Số trang',
    },
    'volumn number': {
        'han_text': '卷號 (Volume number)',
        'han_viet': 'Quyển thứ',
        'pure_vietnamese': 'Số thứ tự quyển'
    },
    'eov': {
        'han_text': '卷终 (End of volume)',
        'han_viet': 'Quyển chung',
        'pure_vietnamese': 'Kết thúc quyển'
}

def get_layout_translations(label):
    """Get translations for layout label"""
    label_lower = label.lower()
    if label_lower in LAYOUT_LABEL_TRANSLATIONS:
        trans = LAYOUT_LABEL_TRANSLATIONS[label_lower]
        return {
            'han_with_english': f"{trans['han_text']}",
            'han_viet': trans['han_viet'],
            'pure_vietnamese': trans['pure_vietnamese']
        }
    else:
        # Default fallback if translation not found
        return {
            'han_with_english': f"({label_lower})",
            'han_viet': label,
            'pure_vietnamese': label
        }

def _has_intersection(ocr_box, layout_box, min_overlap_threshold=0.5):
    """
    Check if two bounding boxes have significant intersection based on Overlap Ratio
    
    Args:
        ocr_box: [x1, y1, x2, y2] coordinates of OCR text box
        layout_box: [x1, y1, x2, y2] coordinates of layout box
        min_overlap_threshold: Minimum overlap ratio threshold (default: 0.5 = 50% of OCR box must be covered)
    
    Returns:
        bool: True if overlap ratio exceeds threshold, False otherwise
    """
    ocr_x1, ocr_y1, ocr_x2, ocr_y2 = ocr_box
    layout_x1, layout_y1, layout_x2, layout_y2 = layout_box
    
    # Check for no intersection first (quick exit)
    if (ocr_x2 <= layout_x1 or ocr_x1 >= layout_x2 or 
        ocr_y2 <= layout_y1 or ocr_y1 >= layout_y2):
        return False
    
    # Calculate intersection area
    intersection_x1 = max(ocr_x1, layout_x1)
    intersection_y1 = max(ocr_y1, layout_y1)
    intersection_x2 = min(ocr_x2, layout_x2)
    intersection_y2 = min(ocr_y2, layout_y2)
    
    intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
    
    # Calculate OCR box area
    ocr_area = (ocr_x2 - ocr_x1) * (ocr_y2 - ocr_y1)
    
    # Avoid division by zero
    if ocr_area <= 0:
        return False
    
    # Calculate overlap ratio (what percentage of OCR box is covered by layout box)
    overlap_ratio = intersection_area / ocr_area
    
    # Primary check: overlap ratio exceeds threshold
    if overlap_ratio >= min_overlap_threshold:
        return True
    
    # Secondary check: If overlap ratio < 50%, check if OCR center is inside layout
    if overlap_ratio < 0.5:
        # Calculate OCR center point
        ocr_center_x = (ocr_x1 + ocr_x2) / 2
        ocr_center_y = (ocr_y1 + ocr_y2) / 2
        
        # Check if center point is inside layout box
        center_inside = (layout_x1 <= ocr_center_x <= layout_x2 and 
                        layout_y1 <= ocr_center_y <= layout_y2)
        
        return center_inside
    
    # If overlap ratio >= 50% but < threshold, reject
    return False

def map_layout_ocr_transliteration_prose_improved(layout_det_res, ocr_result_data, transliteration_data, prose_translation_data, min_overlap_threshold=0.5):
    """
    Map layout detection results with OCR text, transliteration, and prose translation data
    IMPROVED VERSION: Handles multiple layouts of the same type with unique keys and overlap ratio-based intersection
    
    Args:
        layout_det_res: Layout detection results
        ocr_result_data: OCR results with bounding boxes
        transliteration_data: Transliteration results
        prose_translation_data: Prose translation results
        min_overlap_threshold: Minimum overlap ratio threshold for text-to-layout assignment (default: 0.5)
    """
    layout_boxes = []
    label_counts = {}  # Track count of each label type
    
    for i, box_info in enumerate(layout_det_res['res']['boxes']):
        if box_info['label'].lower() != 'page box':
            label = box_info['label']
            
            # Create unique key for layouts of the same type
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
            
            # If there are multiple of the same type, add number
            if label_counts[label] > 1:
                unique_label = f"{label}_{label_counts[label]}"
            else:
                unique_label = label
            
            layout_boxes.append({
                'unique_label': unique_label,
                'original_label': label,
                'bbox': box_info['coordinate'],  # [x1, y1, x2, y2]
                'score': box_info['score'],
                'box_index': i
            })
    
    # Extract OCR results with bounding boxes AND get the correct text order
    ocr_text_list = ocr_result_data['data']['result_ocr_text']  # This has the correct order
    
    ocr_texts = []
    for bbox_info in ocr_result_data['data']['result_bbox']:
        bbox_coords = bbox_info[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        text = bbox_info[1][0]
        confidence = bbox_info[1][1]
        
        # Convert 4-point bbox to standard format [x1, y1, x2, y2]
        x_coords = [point[0] for point in bbox_coords]
        y_coords = [point[1] for point in bbox_coords]
        bbox_standard = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
        
        # Find the correct index in the OCR text list (this preserves the reading order)
        text_order_index = -1
        for i, ocr_text in enumerate(ocr_text_list):
            if ocr_text == text:
                text_order_index = i
                break
        
        ocr_texts.append({
            'text': text,
            'bbox': bbox_standard,
            'confidence': confidence,
            'original_bbox': bbox_coords,
            'text_order_index': text_order_index  # This preserves the OCR text order
        })
    
    # Extract transliteration and prose translation results
    hannom_texts = transliteration_data['data']['result_hannom_text']
    transcriptions = transliteration_data['data']['result_text_transcription']
    prose_translations = prose_translation_data['data']['result']
    
    # Create mapping with all four components
    layout_mapping = {}
    
    for layout in layout_boxes:
        unique_label = layout['unique_label']
        layout_bbox = layout['bbox']
        
        # Find OCR texts that have significant intersection with this layout
        intersecting_ocr = []
        
        for ocr in ocr_texts:
            # Use improved intersection function with overlap ratio threshold
            if _has_intersection(ocr['bbox'], layout_bbox, min_overlap_threshold):
                # Find corresponding transliteration and prose translation
                original_text = ocr['text']
                transcription = ""
                prose_translation = ""
                
                # Match with transliteration data
                for j, hannom_text in enumerate(hannom_texts):
                    if hannom_text == original_text:
                        if j < len(transcriptions):
                            transcription = transcriptions[j]
                        if j < len(prose_translations):
                            prose_translation = prose_translations[j]
                        break
                
                # Calculate overlap ratio for debugging/analysis
                ocr_x1, ocr_y1, ocr_x2, ocr_y2 = ocr['bbox']
                layout_x1, layout_y1, layout_x2, layout_y2 = layout_bbox
                
                intersection_x1 = max(ocr_x1, layout_x1)
                intersection_y1 = max(ocr_y1, layout_y1)
                intersection_x2 = min(ocr_x2, layout_x2)
                intersection_y2 = min(ocr_y2, layout_y2)
                
                intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                ocr_area = (ocr_x2 - ocr_x1) * (ocr_y2 - ocr_y1)
                overlap_ratio = intersection_area / ocr_area if ocr_area > 0 else 0
                
                intersecting_ocr.append({
                    'original_text': original_text,
                    'transcription': transcription,
                    'prose_translation': prose_translation,
                    'bbox': ocr['bbox'],
                    'confidence': ocr['confidence'],
                    'text_order_index': ocr['text_order_index'],
                    'overlap_ratio': overlap_ratio  # For debugging/analysis
                })
        
        layout_mapping[unique_label] = {
            'original_label': layout['original_label'],
            'layout_bbox': layout_bbox,
            'layout_score': layout['score'],
            'box_index': layout['box_index'],
            'ocr_results': intersecting_ocr
        }
    
    return layout_mapping

def process_layout_ocr_mapping(layout_det_res, ocr_result_data, transliteration_data, prose_translation_data, min_overlap_threshold=0.5):
    """
    Simple wrapper function for layout-OCR mapping
    
    Args:
        layout_det_res: Layout detection results
        ocr_result_data: OCR results
        transliteration_data: Transliteration results  
        prose_translation_data: Prose translation results
        min_overlap_threshold: Minimum overlap ratio threshold for text-to-layout assignment (default: 0.5)
    """
    return map_layout_ocr_transliteration_prose_improved(layout_det_res, ocr_result_data, transliteration_data, prose_translation_data, min_overlap_threshold)

def create_improved_prose_layout_summary(mapping_result):
    summary = []
    
    for unique_label, data in mapping_result.items():
        # Sort OCR results by original OCR text order (result_ocr_text)
        sorted_ocr_results = sorted(data['ocr_results'], key=lambda x: x['text_order_index'] if x['text_order_index'] != -1 else float('inf'))
        
        original_texts = []
        transcribed_texts = []
        prose_texts = []
        
        for ocr_item in sorted_ocr_results:
            original_texts.append(ocr_item['original_text'])
            transcribed_texts.append(ocr_item['transcription'])
            prose_texts.append(ocr_item['prose_translation'])
        
        # Get layout label translations
        label_translations = get_layout_translations(data['original_label'])
        
        # Calculate average overlap ratio for this layout (for compatibility with notebook)
        avg_overlap = 0
        if data['ocr_results']:
            avg_overlap = sum(item.get('overlap_ratio', 0) for item in data['ocr_results']) / len(data['ocr_results'])
        
        # Get the minimum text order index for this layout to determine overall order
        min_text_order_index = min([item['text_order_index'] for item in data['ocr_results']]) if data['ocr_results'] else float('inf')
        
        summary.append({
            'layout_label': unique_label,
            'original_label': data['original_label'],
            'label_han_with_english': label_translations['han_with_english'],
            'label_han_viet': label_translations['han_viet'],
            'label_pure_vietnamese': label_translations['pure_vietnamese'],
            'original_combined': ' '.join(original_texts),
            'transcribed_combined': ' '.join(transcribed_texts),
            'prose_combined': ' '.join(prose_texts),
            'original_texts': original_texts,  # List of individual texts
            'transcribed_texts': transcribed_texts,  # List of individual texts
            'prose_texts': prose_texts,  # List of individual texts
            'min_text_order_index': min_text_order_index,
            'box_index': data['box_index'],
            'avg_overlap': avg_overlap,  # For analysis
            'avg_iou': avg_overlap,  # For backward compatibility with notebook and analysis 
            'text_count': len(data['ocr_results'])  # Number of texts assigned
        })
    
    # Sort the entire summary by the minimum text order index of each layout
    summary.sort(key=lambda x: x['min_text_order_index'])
    
    # Remove the temporary min_text_order_index field before returning
    for item in summary:
        del item['min_text_order_index']
    
    return summary

def create_layout_summary(mapping_result):
    """Create organized summary by layout regions - uses improved version"""
    return create_improved_prose_layout_summary(mapping_result)

def analyze_overlap_statistics(mapping_result):
    """
    Analyze overlap ratio statistics from mapping results
    
    Args:
        mapping_result: Results from map_layout_ocr_transliteration_prose_improved
    
    Returns:
        dict: Overlap ratio statistics including overall and per-layout analysis
    """
    all_overlaps = []
    layout_stats = {}
    
    # Collect all overlap ratio values
    for layout_label, data in mapping_result.items():
        overlaps = [item.get('overlap_ratio', 0) for item in data['ocr_results']]
        all_overlaps.extend(overlaps)
        
        if overlaps:
            layout_stats[layout_label] = {
                'count': len(overlaps),
                'min': min(overlaps),
                'max': max(overlaps),
                'avg': sum(overlaps) / len(overlaps)
            }
    
    overall_stats = {}
    if all_overlaps:
        overall_stats = {
            'total_texts': len(all_overlaps),
            'avg_overlap': sum(all_overlaps) / len(all_overlaps),
            'min_overlap': min(all_overlaps),
            'max_overlap': max(all_overlaps),
            'distribution': {
                'low_overlap_count': sum(1 for x in all_overlaps if x < 0.3),
                'medium_overlap_count': sum(1 for x in all_overlaps if 0.3 <= x < 0.7),
                'high_overlap_count': sum(1 for x in all_overlaps if x >= 0.7)
            }
        }
        
        # Add percentages
        total = overall_stats['total_texts']
        overall_stats['distribution']['low_overlap_pct'] = overall_stats['distribution']['low_overlap_count'] / total * 100
        overall_stats['distribution']['medium_overlap_pct'] = overall_stats['distribution']['medium_overlap_count'] / total * 100
        overall_stats['distribution']['high_overlap_pct'] = overall_stats['distribution']['high_overlap_count'] / total * 100
    
    return {
        'overall': overall_stats,
        'by_layout': layout_stats
    }

def suggest_overlap_threshold(mapping_result):
    """
    Suggest optimal overlap ratio threshold based on data distribution
    
    Args:
        mapping_result: Results from map_layout_ocr_transliteration_prose_improved
    
    Returns:
        dict: Suggested thresholds and reasoning
    """
    stats = analyze_overlap_statistics(mapping_result)
    
    if not stats['overall']:
        return {'suggestion': 0.5, 'reason': 'No data available, using default'}
    
    avg_overlap = stats['overall']['avg_overlap']
    min_overlap = stats['overall']['min_overlap']
    
    # Suggest threshold based on average overlap ratio
    if avg_overlap >= 0.8:
        suggestion = 0.7
        reason = 'High average overlap detected, strict threshold recommended'
    elif avg_overlap >= 0.6:
        suggestion = 0.5
        reason = 'Moderate average overlap, default threshold appropriate'
    else:
        suggestion = 0.3
        reason = 'Low average overlap, lenient threshold recommended'
    
    return {
        'suggestion': suggestion,
        'reason': reason,
        'stats_summary': {
            'avg_overlap': avg_overlap,
            'min_overlap': min_overlap,
            'total_texts': stats['overall']['total_texts']
        }
    }
