#!/usr/bin/env python3
"""
Test script for kagglehub model download functionality.
Run this script after installing kagglehub to test the model download.
"""

def test_kagglehub_download():
    """Test the kagglehub model download"""
    try:
        import kagglehub
        print("✅ kagglehub imported successfully")
        
        # Test download (this would actually download the model)
        print("🔄 Testing model download...")
        path = kagglehub.model_download("phuchoangnguyen/model_paddle_layout_nhom_nhan/pyTorch/default")
        print(f"✅ Models downloaded successfully to: {path}")
        
        # Check if the path exists
        import os
        if os.path.exists(path):
            print(f"✅ Download path exists: {path}")
            # List contents
            contents = os.listdir(path)
            print(f"📁 Contents: {contents}")
        else:
            print(f"❌ Download path does not exist: {path}")
            
    except ImportError:
        print("❌ kagglehub is not installed. Please install it with: pip install kagglehub")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_kagglehub_download()
