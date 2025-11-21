# Modern Assyrian GPT API

A FastAPI service for generating Modern Assyrian text using a trained GPT model with SentencePiece tokenization, optimized for NVIDIA GPU acceleration.

## Features

- **SentencePiece Tokenization**: Uses 8,000-piece Modern Assyrian tokenizer
- **GPT Model**: Transformer-based language model trained on Modern Assyrian Old Testament
- **REST API**: FastAPI-based service with automatic documentation
- **GPU Acceleration**: NVIDIA PyTorch container with CUDA 13.0 support
- **Performance Metrics**: Real-time token throughput and latency tracking
- **Health Monitoring**: Extended diagnostics including GPU memory and device info
- **CORS Enabled**: Cross-origin support for web clients

## Project Structure

```
syriac-gpt-api/
├── main.py              # FastAPI application with metrics
├── requirements.txt     # Python dependencies
├── Dockerfile          # NVIDIA PyTorch 25.10 GPU container
├── docker-compose.yml  # Compose config with GPU support
├── docker-entrypoint.sh # CUDA diagnostics on startup
├── test_api.py         # API testing script
├── models/             # Model files (mounted volume)
│   ├── GPTot40.pth     # Trained model weights
│   ├── assyrian_8000.model  # SentencePiece model
│   └── assyrian_8000.vocab  # SentencePiece vocabulary
└── README.md           # This file
```

## Quick Start

### Prerequisites

- Docker with NVIDIA Container Toolkit installed
- NVIDIA GPU with driver ≥ 525.60.13
- Docker Compose v2.0+

### Using Docker Compose (Recommended)

1. **Build and start the GPU-accelerated service:**
   ```bash
   docker compose build
   docker compose up -d
   ```

2. **View startup logs (verify GPU detection):**
   ```bash
   docker logs syriac-gpt-api-syriac-gpt-api-1
   ```
   
   You should see:
   ```
   [entrypoint] torch.cuda.is_available(): True
   [entrypoint] device 0 name: NVIDIA GB10
   Model and tokenizer loaded successfully on cuda
   ```

3. **Test the API:**
   ```bash
   curl http://localhost:8000/health | jq
   ```

### Manual Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the API:**
   ```bash
   python main.py
   ```

## API Usage

### Health Check
```bash
curl http://localhost:8000/health | jq
```

Response includes GPU metrics:
```json
{
  "status": "healthy",
  "device": "cuda",
  "cuda_available": true,
  "cuda_version": "13.0",
  "cuda_device_count": 1,
  "cuda_device_name": "NVIDIA GB10",
  "requests": 5,
  "total_generated_tokens": 200,
  "avg_tokens_per_second": 35.2,
  "last_request_latency": 0.42,
  "last_request_tokens": 40,
  "gpu_memory_allocated": 35759104
}
```

### Performance Metrics
```bash
curl http://localhost:8000/metrics | jq
```

Returns detailed performance stats:
- `requests`: Total generation requests served
- `total_generated_tokens`: Cumulative tokens generated
- `avg_tokens_per_second`: Sustained decode throughput
- `last_request_latency`: Wall time for last request (seconds)
- `gpu_memory_allocated`: Current GPU memory usage (bytes)

### Generate Text
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "ܐܲܒ݂ܪܵܗܵܡ",
       "max_new_tokens": 50,
       "temperature": 0.8,
       "top_k": 40
     }'
```

### API Documentation
Visit `http://localhost:8000/docs` for interactive Swagger UI documentation.

## Parameters

- **prompt**: Input Modern Assyrian text to continue from
- **max_new_tokens**: Maximum number of tokens to generate (default: 50)
- **temperature**: Sampling temperature (0.1-2.0, default: 1.0)
- **top_k**: Top-k sampling parameter (optional)

## Model Details

- **Architecture**: GPT with 3 layers, 4 attention heads, 256 embedding dimensions
- **Vocabulary**: 8,000 SentencePiece tokens
- **Training**: 40 epochs on Modern Assyrian Old Testament text
- **Context Length**: 128 tokens

## Docker Configuration

The service uses NVIDIA's official PyTorch container (`nvcr.io/nvidia/pytorch:25.10-py3`) with:
- PyTorch 2.9.0a0 (nightly build)
- CUDA 13.0 runtime
- GPU auto-detection via `gpus: all` in docker-compose.yml
- IPC and ulimits optimized for NVIDIA workloads

### GPU Requirements

Verified on:
- **Hardware**: NVIDIA GB10 (compute capability 12.1)
- **Driver**: 580.95.05 or newer
- **CUDA**: 13.0+ (provided by container)

### Environment Variables

Configure in `docker-compose.yml`:
```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=all       # Expose all GPUs
  - NVIDIA_DRIVER_CAPABILITIES=compute,utility
  - PYTHONUNBUFFERED=1
```

### CPU Fallback

If GPU unavailable, the service auto-falls back to CPU. Check device in `/health` response.

## Testing

### Basic Functionality
```bash
python test_api.py
```

### Benchmark Performance
Run 10 requests and measure throughput:
```bash
PROMPT='ܫܠܡܐ'
REQUESTS=10
TOKENS=40
for i in $(seq 1 $REQUESTS); do
  START=$(date +%s.%N)
  curl -s -X POST http://localhost:8000/generate \
    -H 'Content-Type: application/json' \
    -d "{\"prompt\":\"$PROMPT\",\"max_new_tokens\":$TOKENS}" >/dev/null
  END=$(date +%s.%N)
  awk -v s=$START -v e=$END 'BEGIN{printf("Req %02d latency: %.3fs\n",'$i', e-s)}'
done

echo "Final metrics:"
curl -s http://localhost:8000/metrics | jq
```

Expected GPU performance: **30-40 tokens/second** (NVIDIA GB10).

### Cross-Device Testing (LAN)
From another device on the same network:
```bash
# Replace with your host IP
curl http://10.0.0.205:8000/health | jq
curl -X POST http://10.0.0.205:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"ܫܠܡܐ","max_new_tokens":30}' | jq
```

## Development

### Adding New Endpoints

Edit `main.py` to add new FastAPI routes.

### Model Updates

1. Train new model weights
2. Save as `GPTot{epoch}.pth`
3. Update the model path in `main.py`
4. Rebuild the Docker image

### Tokenizer Updates

1. Train new SentencePiece model
2. Update model and vocab files in `models/`
3. Update vocab_size in Config class if changed

## Troubleshooting

### Container Startup Issues
```bash
# Check container status
docker compose ps

# View logs
docker compose logs -f

# Restart
docker compose down && docker compose up -d
```

### GPU Not Detected
1. Verify NVIDIA runtime:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
   ```
2. Check driver version: `nvidia-smi`
3. Reinstall NVIDIA Container Toolkit if needed

### CUDA Kernel Errors
If you see "no kernel image available for execution":
- Your GPU compute capability may not be supported by the PyTorch wheel
- Check `[entrypoint]` logs for device capability
- Consider building PyTorch from source with `TORCH_CUDA_ARCH_LIST=12.1`

### Performance Issues
- Monitor GPU usage: `nvidia-smi -l 1`
- Check memory: `curl localhost:8000/metrics | jq .gpu_memory_allocated`
- Reduce batch size or context length if OOM
- Verify IPC mode in compose: `ipc: host`

### Model Loading Issues
- Ensure model files exist in `models/` directory
- Check file permissions
- Verify PyTorch version compatibility

### API Connection Issues
- Verify service is running: `docker compose ps`
- Check logs: `docker compose logs`
- Ensure port 8000 is not blocked by firewall
- For LAN access, bind to `0.0.0.0` (already configured)

## License

This project is part of the Modern Assyrian language model research.