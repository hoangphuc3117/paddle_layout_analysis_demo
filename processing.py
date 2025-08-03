import requests
from utils import BASE_API_URL

def _has_intersection(ocr_box, layout_box):
    """Check if two bounding boxes have any intersection"""
    ocr_x1, ocr_y1, ocr_x2, ocr_y2 = ocr_box
    layout_x1, layout_y1, layout_x2, layout_y2 = layout_box
    
    if (ocr_x2 <= layout_x1 or ocr_x1 >= layout_x2 or 
        ocr_y2 <= layout_y1 or ocr_y1 >= layout_y2):
        return False
    return True

# IMPROVED MAPPING FUNCTION - XỬ LÝ NHIỀU LAYOUT CÙNG LOẠI
def map_layout_ocr_transliteration_prose_improved(layout_det_res, ocr_result_data, transliteration_data, prose_translation_data):
    """
    Map layout detection results with OCR text, transliteration, and prose translation data
    IMPROVED VERSION: Handles multiple layouts of the same type (e.g., multiple bibliography boxes)
    
    Args:
        layout_det_res: Layout detection results
        ocr_result_data: OCR results from API
        transliteration_data: Transliteration results from API
        prose_translation_data: Prose translation results from API
        
    Returns:
        dict: Comprehensive mapping with unique keys for each layout box
    """
    
    # Extract layout boxes (exclude page box) with unique identifiers
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
        
        # Find OCR texts that intersect with this layout
        intersecting_ocr = []
        
        for ocr in ocr_texts:
            if _has_intersection(ocr['bbox'], layout_bbox):
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
                
                intersecting_ocr.append({
                    'original_text': original_text,
                    'transcription': transcription,
                    'prose_translation': prose_translation,
                    'bbox': ocr['bbox'],
                    'confidence': ocr['confidence'],
                    'text_order_index': ocr['text_order_index']
                })
        
        layout_mapping[unique_label] = {
            'original_label': layout['original_label'],
            'layout_bbox': layout_bbox,
            'layout_score': layout['score'],
            'box_index': layout['box_index'],
            'ocr_results': intersecting_ocr
        }
    
    return layout_mapping

def sort_ocr_by_text_order(ocr_results):
    """Sort OCR results by their original text order index"""
    return sorted(ocr_results, key=lambda x: x['text_order_index'] if x['text_order_index'] != -1 else float('inf'))

def process_layout_ocr_mapping(layout_det_res, ocr_result_data, transliteration_data, prose_translation_data):
    """
    Complete processing: map layout -> OCR -> transliteration -> prose translation
    Returns organized results by layout regions
    Uses the improved mapping function for better handling of multiple layouts
    """
    return map_layout_ocr_transliteration_prose_improved(layout_det_res, ocr_result_data, transliteration_data, prose_translation_data)

def create_improved_prose_layout_summary(mapping_result):
    """Create a summary of all text types organized by layout, maintaining original OCR text order"""
    summary = []
    
    for unique_label, data in mapping_result.items():
        # Sort OCR results by original OCR text order
        sorted_ocr_results = sort_ocr_by_text_order(data['ocr_results'])
        
        original_texts = []
        transcribed_texts = []
        prose_texts = []
        
        for ocr_item in sorted_ocr_results:
            original_texts.append(ocr_item['original_text'])
            transcribed_texts.append(ocr_item['transcription'])
            prose_texts.append(ocr_item['prose_translation'])
        
        # Get the minimum text order index for this layout to determine overall order
        min_text_order_index = min([item['text_order_index'] for item in data['ocr_results']]) if data['ocr_results'] else float('inf')
        
        summary.append({
            'layout_label': unique_label,
            'original_label': data['original_label'],
            'original_combined': ' '.join(original_texts),
            'transcribed_combined': ' '.join(transcribed_texts),
            'prose_combined': ' '.join(prose_texts),
            'min_text_order_index': min_text_order_index,
            'box_index': data['box_index']
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
