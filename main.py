from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from pathlib import Path
import math
import time
from typing import Optional

app = FastAPI(title="Modern Assyrian GPT API", description="API for generating Modern Assyrian text using GPT model")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model Configuration
class Config:
    def __init__(self):
        self.n_layer = 3
        self.n_head = 4
        self.n_embd = 256
        self.vocab_size = 8000  # SentencePiece vocab size
        self.block_size = 128
        self.embd_pdrop = 0.1
        self.resid_pdrop = 0.1
        self.attn_pdrop = 0.1

# Model Architecture Components
class GELU(nn.Module):
    def forward(self, x):
        return 0.5*x*(1.0+torch.tanh(math.sqrt(2.0/math.pi)*\
                       (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.register_buffer("bias", torch.tril(torch.ones(\
                   config.block_size, config.block_size))
             .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        hs = C // self.n_head
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) *\
            (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, \
                              float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj = nn.Linear(4 * config.n_embd, config.n_embd),
            act    = GELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf=lambda x:m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config)
                               for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),))
        self.lm_head = nn.Linear(config.n_embd,
                                 config.vocab_size, bias=False)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0,
                  std=0.02/math.sqrt(2 * config.n_layer))
    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(0,t,dtype=torch.long).unsqueeze(0).to(device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

# Global variables for model and tokenizer
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Metrics counters
generation_requests = 0
total_generated_tokens = 0
total_generation_time = 0.0
last_request_latency = 0.0
last_request_tokens = 0

def load_model_and_tokenizer():
    global model, tokenizer

    # Load tokenizer
    tokenizer_model_path = Path("models/assyrian_8000.model")
    if not tokenizer_model_path.exists():
        raise FileNotFoundError(f"Tokenizer model not found at {tokenizer_model_path}")

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(tokenizer_model_path))

    # Load model
    config = Config()
    config.vocab_size = tokenizer.get_piece_size()

    model = Model(config)
    model_path = "models/GPTot40.pth"
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Model and tokenizer loaded successfully on {device}")

# Sampling function
def sample(idx, max_new_tokens, temperature=1.0, top_k=None):
    model.eval()
    original_length = len(idx[0])

    for _ in range(max_new_tokens):
        if idx.size(1) <= model.block_size:
            idx_cond = idx
        else:
            idx_cond = idx[:, -model.block_size:]

        logits = model(idx_cond.to(device))
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next.cpu()), dim=1)

    return idx[:, original_length:]

# Pydantic models for API
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 1.0
    top_k: Optional[int] = None

class GenerateResponse(BaseModel):
    generated_text: str
    prompt: str
    parameters: dict

# API Endpoints
@app.on_event("startup")
async def startup_event():
    # Basic CUDA diagnostics before loading
    try:
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {getattr(torch.version, 'cuda', None)}")
        try:
            print(f"CUDA device count: {torch.cuda.device_count()}")
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                print(f"CUDA device 0 name: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"CUDA device query error: {e}")
    except Exception as e:
        print(f"CUDA diagnostics failed: {e}")

    load_model_and_tokenizer()

@app.get("/")
async def root():
    return {"message": "Modern Assyrian GPT API", "status": "running"}

@app.get("/health")
async def health():
    # Include extended CUDA diagnostics
    try:
        cuda_available = torch.cuda.is_available()
        cuda_version = getattr(torch.version, 'cuda', None)
        cuda_device_count = torch.cuda.device_count()
        cuda_device_name = None
        if cuda_available and cuda_device_count > 0:
            cuda_device_name = torch.cuda.get_device_name(0)
        return {
            "status": "healthy",
            "device": device,
            "cuda_available": cuda_available,
            "cuda_version": cuda_version,
            "cuda_device_count": cuda_device_count,
            "cuda_device_name": cuda_device_name,
            "requests": generation_requests,
            "total_generated_tokens": total_generated_tokens,
            "avg_tokens_per_second": (total_generated_tokens / total_generation_time) if total_generation_time > 0 else 0.0,
            "last_request_latency": last_request_latency,
            "last_request_tokens": last_request_tokens,
            "gpu_memory_allocated": torch.cuda.memory_allocated(0) if cuda_available and cuda_device_count > 0 else 0,
        }
    except Exception as e:
        return {"status": "healthy", "device": device, "cuda_error": str(e)}

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    try:
        global generation_requests, total_generated_tokens, total_generation_time, last_request_latency, last_request_tokens
        start_time = time.perf_counter()
        # Encode prompt
        prompt_tokens = tokenizer.encode(request.prompt)
        idx = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0)

        # Generate new tokens
        new_tokens = sample(idx, request.max_new_tokens, request.temperature, request.top_k)

        # Decode generated tokens
        generated_ids = prompt_tokens + new_tokens.squeeze(0).tolist()
        generated_text = tokenizer.decode(generated_ids)

        # Metrics update
        end_time = time.perf_counter()
        latency = end_time - start_time
        tokens_generated = len(generated_ids) - len(prompt_tokens)
        generation_requests += 1
        total_generated_tokens += tokens_generated
        total_generation_time += latency
        last_request_latency = latency
        last_request_tokens = tokens_generated

        return GenerateResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            parameters={
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "top_k": request.top_k
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count()
    return {
        "device": device,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_version": getattr(torch.version, 'cuda', None),
        "requests": generation_requests,
        "total_generated_tokens": total_generated_tokens,
        "total_generation_time": total_generation_time,
        "avg_tokens_per_second": (total_generated_tokens / total_generation_time) if total_generation_time > 0 else 0.0,
        "last_request_latency": last_request_latency,
        "last_request_tokens": last_request_tokens,
        "gpu_memory_allocated": torch.cuda.memory_allocated(0) if cuda_available and cuda_device_count > 0 else 0,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)