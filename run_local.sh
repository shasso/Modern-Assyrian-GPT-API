#!/bin/bash
# Script to run Syriac GPT API locally with GPU support

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Syriac GPT API (Local GPU Mode)${NC}"
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import torch" 2>/dev/null; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install -r requirements.txt
fi

# Check CUDA availability
echo -e "${GREEN}Checking CUDA availability...${NC}"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else \"CPU\"}')"
echo ""

# Check if model files exist
if [ ! -f "models/GPTot40.pth" ]; then
    echo -e "${RED}Error: Model file not found at models/GPTot40.pth${NC}"
    exit 1
fi

if [ ! -f "models/assyrian_8000.model" ]; then
    echo -e "${RED}Error: Tokenizer file not found at models/assyrian_8000.model${NC}"
    exit 1
fi

echo -e "${GREEN}Starting API server on http://localhost:8000${NC}"
echo -e "${YELLOW}Press CTRL+C to stop${NC}"
echo ""

# Run the application
python main.py
