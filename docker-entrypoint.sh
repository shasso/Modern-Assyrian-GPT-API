#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] Python version: $(python -V)"
echo "[entrypoint] Torch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'torch not found')"
python - <<'PY'
import torch, os
print('[entrypoint] CUDA visible devices env:', os.getenv('NVIDIA_VISIBLE_DEVICES'))
print('[entrypoint] torch.cuda.is_available():', torch.cuda.is_available())
print('[entrypoint] torch.version.cuda:', torch.version.cuda)
if torch.cuda.is_available():
    try:
        print('[entrypoint] device count:', torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f'[entrypoint] device {i} name:', torch.cuda.get_device_name(i))
            print(f'[entrypoint] device {i} capability:', torch.cuda.get_device_capability(i))
    except Exception as e:
        print('[entrypoint] Error querying CUDA devices:', e)
else:
    print('[entrypoint] No CUDA devices available inside container.')
PY

exec "$@"
