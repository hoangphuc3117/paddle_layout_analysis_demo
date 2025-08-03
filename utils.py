import os
import streamlit as st
import kagglehub
import requests
from pathlib import Path
import json
import concurrent.futures
import threading

BASE_API_URL = 'https://kimhannom.clc.hcmus.edu.vn'

def download_models_from_kaggle():
    """Download models from Kaggle Hub"""
    try:
        with st.spinner("Downloading models from Kaggle Hub... This may take a few minutes."):
            # Download latest version
            path = kagglehub.model_download("phuchoangnguyen/model_paddle_layout_nhom_nhan/pyTorch/default")
        return path
    except Exception as e:
        return None

def get_model_paths():
    """Download models if not already downloaded and return the paths."""
    if 'kaggle_model_path' not in st.session_state:
        st.session_state.kaggle_model_path = download_models_from_kaggle()
    
    kaggle_model_path = st.session_state.kaggle_model_path
    
    if not kaggle_model_path or not os.path.exists(kaggle_model_path):
        st.error("Failed to download models from Kaggle Hub. Please check your internet connection and Kaggle credentials.")
        st.stop()

    # Construct paths to the models within the downloaded directory
    layout_detection_dir = os.path.join(kaggle_model_path, "models/layout_detection")
    text_detection_dir = os.path.join(kaggle_model_path, "models/text_detection")
    text_recognition_dir = os.path.join(kaggle_model_path, "models/text_recognition")

    # Verify that the model directories exist
    if not all(os.path.exists(p) for p in [layout_detection_dir, text_detection_dir, text_recognition_dir]):
        st.error(f"Model directories not found after download. Please check the structure of the downloaded model files in '{kaggle_model_path}'.")
        st.stop()
        
    return layout_detection_dir, text_detection_dir, text_recognition_dir

def upload_image_to_server(image_data, file_name, timeout=30):
    try:
        files = {'image_file': (file_name, image_data, 'image/*')}
        
        headers = {
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Origin': 'https://kimhannom.clc.hcmus.edu.vn',
            'Referer': 'https://kimhannom.clc.hcmus.edu.vn/',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0',
            'sec-ch-ua': '"Not)A;Brand";v="8", "Chromium";v="138", "Microsoft Edge";v="138"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"'
        }
        response = requests.post(
            f'{BASE_API_URL}/api/web/clc-sinonom/image-upload',
            files=files,
            timeout=timeout,
            headers=headers,
            verify=False
        )
        
        response.raise_for_status()  # Raises HTTPError for bad responses
        
        return response
            
    except requests.exceptions.RequestException as e:
        print(f"Upload failed: {e}")
        raise

def image_ocr(image_name, ocr_id):
    headers = {
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Origin': 'https://kimhannom.clc.hcmus.edu.vn',
        'Referer': 'https://kimhannom.clc.hcmus.edu.vn/',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0',
        'sec-ch-ua': '"Not)A;Brand";v="8", "Chromium";v="138", "Microsoft Edge";v="138"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"'
    }
    response = requests.post(f'{BASE_API_URL}/api/web/clc-sinonom/image-ocr', json={"ocr_id":ocr_id,"file_name":image_name},  headers=headers, verify=False)
    return response

def sinonom_transliteration(text, font_type, lang_type):
    headers = {
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Origin': 'https://kimhannom.clc.hcmus.edu.vn',
        'Referer': 'https://kimhannom.clc.hcmus.edu.vn/',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0',
        'sec-ch-ua': '"Not)A;Brand";v="8", "Chromium";v="138", "Microsoft Edge";v="138"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"'
    }
    response = requests.post(f'{BASE_API_URL}/api/web/clc-sinonom/sinonom-transliteration', json={"text": text, "lang_type": lang_type, "font_type": font_type}, headers=headers, verify=False)
    return response

def image_ocr(image_name, font_type, lang_type, ocr_id, reading_direction):
    headers = {
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Origin': 'https://kimhannom.clc.hcmus.edu.vn',
        'Referer': 'https://kimhannom.clc.hcmus.edu.vn/',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0',
        'sec-ch-ua': '"Not)A;Brand";v="8", "Chromium";v="138", "Microsoft Edge";v="138"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"'
    }
    response = requests.post(f'{BASE_API_URL}/api/web/clc-sinonom/image-ocr', json={"ocr_id":ocr_id,"file_name":image_name,"font_type":font_type,"lang_type":lang_type,"reading_direction":reading_direction},  headers=headers, verify=False)
    return response

def sinonom_transliteration(text, font_type, lang_type):
    headers = {
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Origin': 'https://kimhannom.clc.hcmus.edu.vn',
        'Referer': 'https://kimhannom.clc.hcmus.edu.vn/',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0',
        'sec-ch-ua': '"Not)A;Brand";v="8", "Chromium";v="138", "Microsoft Edge";v="138"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"'
    }
    response = requests.post(f'{BASE_API_URL}/api/web/clc-sinonom/sinonom-transliteration', json={"text": text, "lang_type": lang_type, "font_type": font_type}, headers=headers, verify=False)
    return response

def sinonom_prose_translation(text, lang_type=1):
    """Call prose translation API"""
    headers = {
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Origin': 'https://kimhannom.clc.hcmus.edu.vn',
        'Referer': 'https://kimhannom.clc.hcmus.edu.vn/',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0',
        'Content-Type': 'application/json'
    }
    response = requests.post(f'{BASE_API_URL}/api/web/clc-sinonom/sinonom-fair-seq',
                           json={"text": text, "lang_type": lang_type},
                           headers=headers, verify=False)
    return response