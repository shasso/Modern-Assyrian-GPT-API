# Running Syriac GPT API Locally (Without Docker)

This guide shows how to run the API directly on your host with GPU support.

## Prerequisites

- Python 3.12
- CUDA-capable GPU (tested with NVIDIA GB10)
- NVIDIA drivers installed
- Model files: `models/GPTot40.pth` and `models/assyrian_8000.model`

## Quick Start

**Recommended**: Use CPU mode for stable operation:

```bash
./run_local_cpu.sh
```

This script will:
- Create a virtual environment (if needed)
- Install PyTorch with CUDA 12.4 support
- Install all dependencies
- Force CPU mode (due to GB10 GPU compatibility issues)
- Start the API server on port 8000

### Alternative: Attempt GPU Mode

```bash
./run_local.sh
```

Note: This will attempt GPU execution but will fail on NVIDIA GB10 due to unsupported compute capability.

If you prefer manual setup:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install -r requirements.txt

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Run the API
python main.py
```

## Verify It's Running

```bash
# Check health
curl http://localhost:8000/health

# Generate text
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "ܡܪܝܐ",
    "max_new_tokens": 50,
    "temperature": 0.8,
    "top_k": 40
  }'
```

## GPU Support Status

**Current Status**: NVIDIA GB10 GPU (compute capability sm_121) is **not supported** by prebuilt PyTorch wheels.

While `torch.cuda.is_available()` returns `True` and detects the GPU, PyTorch cannot execute CUDA kernels on this device. You'll see this error during inference:
```
CUDA error: no kernel image is available for execution on the device
```

**Workaround**: Run in CPU mode by setting `CUDA_VISIBLE_DEVICES=""`:

```bash
# Force CPU mode
./run_local_cpu.sh

# Or manually:
export CUDA_VISIBLE_DEVICES=""
python main.py
```

To check device status in Python:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
```

**Future GPU Support**: Wait for PyTorch to add sm_121 support, or build PyTorch from source with CUDA 13.0 support.

## Troubleshooting

### Virtual Environment Issues
If you get permission errors, ensure you're in the project directory:
```bash
cd /home/lamadu/jupyterlab/syriac-gpt-api
```

### CUDA Not Available
- Verify NVIDIA drivers: `nvidia-smi`
- Check PyTorch installation: `python -c "import torch; print(torch.version.cuda)"`
- Reinstall PyTorch with CUDA: `pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124`

### Missing Model Files
Ensure model files are in the `models/` directory:
- `models/GPTot40.pth`
- `models/assyrian_8000.model`
- `models/assyrian_8000.vocab`

### Port Already in Use
If port 8000 is occupied, edit `main.py` and change the port in the last line, or stop other services using port 8000.

## Stopping the Server

Press `CTRL+C` in the terminal where the server is running.

## Performance Notes

- **GPU Mode**: Inference will use CUDA if properly supported
- **CPU Mode**: Automatic fallback if GPU compatibility issues occur
- **Memory**: The model requires ~500MB of RAM/VRAM

## API Endpoints

- `GET /` - API info
- `GET /health` - Health check with device info
- `POST /generate` - Generate Syriac text
