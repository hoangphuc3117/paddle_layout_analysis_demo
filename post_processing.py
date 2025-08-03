def _has_intersection(ocr_box, layout_box):
    """Check if two bounding boxes have any intersection"""
    ocr_x1, ocr_y1, ocr_x2, ocr_y2 = ocr_box
    layout_x1, layout_y1, layout_x2, layout_y2 = layout_box
    
    # Check for no intersection
    if (ocr_x2 <= layout_x1 or ocr_x1 >= layout_x2 or 
        ocr_y2 <= layout_y1 or ocr_y1 >= layout_y2):
        return False
    
    return True

def map_ocr_to_layout_intersection(text_results, layout_results):
    """Map OCR text results to layout regions using intersection detection"""
    mapping = []
    total_texts = len(text_results)
    mapped_count = 0
    
    for text_data in text_results:
        # Handle different data structures
        if isinstance(text_data, dict):
            # Standard format: {'bbox': [...], 'text': '...', 'confidence': ...}
            if 'bbox' in text_data:
                text_box = text_data['bbox']
                text = text_data.get('text', '')
                confidence = text_data.get('confidence', 0.0)
            # Alternative format: {'box': [...], 'text': '...', 'score': ...}
            elif 'box' in text_data:
                text_box = text_data['box']
                text = text_data.get('text', '')
                confidence = text_data.get('score', 0.0)
            else:
                print(f"⚠️ Unknown text_data format: {text_data}")
                continue
        else:
            print(f"⚠️ Expected dict but got {type(text_data)}: {text_data}")
            continue
        
        mapped_layouts = []
        for layout_data in layout_results:
            layout_box = layout_data['bbox']
            layout_label = layout_data['label']
            
            # Skip page_box layout
            if layout_label.lower() == 'page_box':
                continue
                
            if _has_intersection(text_box, layout_box):
                mapped_layouts.append({
                    'layout_label': layout_label,
                    'layout_bbox': layout_box
                })
        
        mapping.append({
            'text': text,
            'text_bbox': text_box,
            'confidence': confidence,
            'mapped_layouts': mapped_layouts
        })
        
        if mapped_layouts:
            mapped_count += 1
    
    success_rate = (mapped_count / total_texts) * 100 if total_texts > 0 else 0
    
    return mapping, success_rate

def group_texts_by_layout(mapping_result):
    """Group texts by their layout regions"""
    layout_groups = {}
    
    for item in mapping_result:
        text = item['text']
        mapped_layouts = item['mapped_layouts']
        
        # If text is mapped to layouts
        if mapped_layouts:
            for layout in mapped_layouts:
                label = layout['layout_label']
                if label not in layout_groups:
                    layout_groups[label] = []
                layout_groups[label].append(text)
        else:
            # Text not mapped to any layout
            if 'unmapped' not in layout_groups:
                layout_groups['unmapped'] = []
            layout_groups['unmapped'].append(text)
    
    return layout_groups