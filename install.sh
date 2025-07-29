#!/bin/bash

# Installation script for PP-Structure V3 Demo with Kaggle Hub support

echo "🚀 Setting up PP-Structure V3 Demo with Kaggle Hub support..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10+ first."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Create virtual environment (optional but recommended)
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Test kagglehub installation
echo "🧪 Testing kagglehub installation..."
python3 test_kagglehub.py

echo ""
echo "🎉 Installation complete!"
echo ""
echo "To start the application:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Run the app: streamlit run web_demo.py"
echo ""
echo "💡 Tips:"
echo "  - Use the 'Download Models from Kaggle' button in the web interface"
echo "  - Check the README.md for detailed usage instructions"
echo "  - For deployment, all dependencies are already listed in requirements.txt"
