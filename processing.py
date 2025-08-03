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

def process_layout_ocr_mapping(layout_det_res, ocr_result_data, transliteration_data, prose_translation_data):
    """
    Complete processing: map layout -> OCR -> transliteration -> prose translation
    Returns organized results by layout regions
    """
    # Extract layout boxes (exclude page box)
    layout_boxes = []
    for box_info in layout_det_res['res']['boxes']:
        if box_info['label'].lower() != 'page box':
            layout_boxes.append({
                'label': box_info['label'],
                'bbox': box_info['coordinate'],
                'score': box_info['score']
            })
    
    # Extract OCR results with proper text order
    ocr_text_list = ocr_result_data['data']['result_ocr_text']
    ocr_texts = []
    
    for bbox_info in ocr_result_data['data']['result_bbox']:
        bbox_coords = bbox_info[0]
        text = bbox_info[1][0]
        confidence = bbox_info[1][1]
        
        # Convert to standard bbox format
        x_coords = [point[0] for point in bbox_coords]
        y_coords = [point[1] for point in bbox_coords]
        bbox_standard = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
        
        # Find text order index
        text_order_index = next((i for i, ocr_text in enumerate(ocr_text_list) if ocr_text == text), -1)
        
        ocr_texts.append({
            'text': text,
            'bbox': bbox_standard,
            'confidence': confidence,
            'text_order_index': text_order_index
        })
    
    # Extract translation results
    hannom_texts = transliteration_data['data']['result_hannom_text']
    transcriptions = transliteration_data['data']['result_text_transcription']
    prose_translations = prose_translation_data['data']['result']
    
    # Create mapping
    layout_mapping = {}
    
    for layout in layout_boxes:
        layout_label = layout['label']
        layout_bbox = layout['bbox']
        intersecting_ocr = []
        
        for ocr in ocr_texts:
            if _has_intersection(ocr['bbox'], layout_bbox):
                original_text = ocr['text']
                transcription = ""
                prose_translation = ""
                
                # Find matching translations
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
        
        layout_mapping[layout_label] = {
            'layout_bbox': layout_bbox,
            'layout_score': layout['score'],
            'ocr_results': intersecting_ocr
        }
    
    return layout_mapping

def create_layout_summary(mapping_result):
    """Create organized summary by layout regions"""
    summary = []
    
    for layout_label, data in mapping_result.items():
        # Sort by text order
        sorted_ocr_results = sorted(data['ocr_results'], key=lambda x: x['text_order_index'])
        
        original_texts = [item['original_text'] for item in sorted_ocr_results]
        transcribed_texts = [item['transcription'] for item in sorted_ocr_results]
        prose_texts = [item['prose_translation'] for item in sorted_ocr_results]
        
        min_order_index = min([item['text_order_index'] for item in data['ocr_results']]) if data['ocr_results'] else float('inf')
        
        summary.append({
            'layout_label': layout_label,
            'original_combined': ' '.join(original_texts),
            'transcribed_combined': ' '.join(transcribed_texts),
            'prose_combined': ' '.join(prose_texts),
            'min_order_index': min_order_index
        })
    
    # Sort by reading order
    summary.sort(key=lambda x: x['min_order_index'])
    
    # Clean up
    for item in summary:
        del item['min_order_index']
    
    return summary
