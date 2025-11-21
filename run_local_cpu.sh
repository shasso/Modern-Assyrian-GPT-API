#!/bin/bash
# Script to run Modern Assyrian GPT API locally
# Force CPU mode due to GB10 GPU compatibility issues with PyTorch

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Modern Assyrian GPT API (Local Mode)${NC}"
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

# Check device availability
echo -e "${GREEN}Checking device availability...${NC}"
python -c "
import torch
cuda_available = torch.cuda.is_available()
print(f'CUDA available: {cuda_available}')
if cuda_available:
    print(f'Device count: {torch.cuda.device_count()}')
    if torch.cuda.device_count() > 0:
        print(f'Device: {torch.cuda.get_device_name(0)}')
        print(f'Compute capability: {torch.cuda.get_device_capability(0)}')
        print('')
        print('NOTE: NVIDIA GB10 (sm_121) is not supported by current PyTorch.')
        print('      Forcing CPU mode for compatibility.')
else:
    print('Using CPU mode')
"
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

# Force CPU mode to avoid CUDA kernel errors on unsupported GPU
export CUDA_VISIBLE_DEVICES=""

# Run the application
python main.py
