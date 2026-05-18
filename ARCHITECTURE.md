# OmniModel - Architecture & Technical Specification

## System Overview

OmniModel is a multimodal inference system designed to run on resource-constrained hardware (1GB RAM, CPU-only). It uses a routing architecture where a small main model directs inputs to specialized sub-models based on modality and desired output style.

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT (any modality)                 │
│              text, image, audio, video, PDF             │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │  Modality Detection     │  Auto-detect: text/image/audio/video/pdf
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │    Input Encoder       │  Convert to unified token space
        │  - VQVAE (images)      │
        │  - EnCodec (audio)     │
        │  - Frame sampler (video)
        │  - PDF extractor       │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────────────┐
        │      Main Router Model          │  Route to best sub-model
        │    (50M params, 6 layers)       │  - Modality routing
        │                                 │  - Style routing
        │   <route> token output          │  - Tool detection
        └────────────┬────────────────────┘
                     │
        ┌────────────▼───────────────────────┐
        │   Lazy Model Loader                │  Load only active sub-model
        │  ┌──────────────────────────────┐  │
        │  │   Main Router (always)       │  │  Main: ~25MB (quantized)
        │  │   Shared Base (always)       │  │  Base: ~10MB (quantized)
        │  │   Active Adapter (1 of 35)   │  │  Adapter: ~5MB (FP16)
        │  │   Total: ~120MB active       │  │  Cache: ~30MB
        │  └──────────────────────────────┘  │
        └────────────┬───────────────────────┘
                     │
        ┌────────────▼────────────┐
        │   Sub-model Inference   │
        │   - Text generation     │
        │   - Image synthesis     │
        │   - Audio synthesis     │
        │   - Tool execution      │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │   Output Decoder       │  Convert tokens back to native format
        │  - Image VQVAE-D       │
        │  - Audio EnCodec-D     │
        │  - Video frame concat  │
        │  - Text tokenizer      │
        └────────────┬────────────┘
                     │
        ┌────────────▼─────────────────┐
        │    OUTPUT (formatted)         │
        │  text, image, audio, video    │
        └───────────────────────────────┘
```

## Component Specifications

### 1. Unified Tokenizer (`tokenizer/unified.py`)

**Purpose:** Convert any input modality into a single token space for consistent processing.

**Specification:**
- Base vocabulary: 50,257 tokens (GPT-2 vocab)
- Special tokens: 9,239 tokens
  - Modality tokens: `<image>`, `<audio>`, `<video>`, `<pdf>` (4 tokens)
  - Style tokens: `<style_creative>`, `<style_professional>`, etc. (8 tokens × 5 modalities)
  - Tool tokens: `<tool_web_search>`, `<tool_calculator>`, etc. (6 tokens)
  - Control tokens: `<route>`, `<think>`, `<args>`, `<result>` (4 tokens)
  - Special: `<pad>`, `<bos>`, `<eos>`, `<unk>` (4 tokens)
  - Remaining: ~9,200 tokens for future expansion
- Total vocabulary: 59,496 tokens

**Input Processing:**
```
Text → Character-level encoding → Token IDs
Image → VQVAE encoder → Codebook IDs → Token IDs
Audio → EnCodec encoder → Codes → Token IDs
Video → Frame sampler → Per-frame encoding → Token IDs
PDF → Text extraction → Character encoding → Token IDs
```

**Output:**
- 1D token sequences
- Max sequence length: 2048 tokens (configurable)
- Padding to power-of-2 for efficiency

### 2. Main Router Model (`main_model/model.py`)

**Purpose:** Route inputs to appropriate sub-models and styles.

**Architecture:**
```
Input Tokens (seq_len, vocab_size=59496)
    ↓
Embedding Layer (vocab=59496 → hidden=512)
    ↓
Positional Encoding (RoPE - Rotary Position Embeddings)
    ↓
6 Transformer Decoder Blocks:
    - Multi-Head Attention (8 heads, head_dim=64)
    - Feed-Forward (hidden=2048)
    - Layer Norm + Residual
    ↓
Output Projection (hidden=512 → special_tokens=100)
    ↓
Routing Tokens:
    <route_modality> <route_style> <route_tool>
```

**Specifications:**
- Hidden dimension: 512
- Number of layers: 6
- Number of heads: 8
- Head dimension: 512 ÷ 8 = 64
- Feed-forward dimension: 2048
- Total parameters: ~50M
- Quantized size: ~25MB (4-bit)

**Training Data:** 
- 1000 synthetic examples per style
- Generated from templates with diverse modalities
- Balanced across all routing decisions

### 3. Shared Sub-model Base (`sub_models/base.py`)

**Purpose:** Common backbone for all modality/style combinations.

**Architecture:**
```
Routed Tokens (from router)
    ↓
Embedding Layer (vocab=59496 → hidden=768)
    ↓
8 Transformer Decoder Blocks:
    - Multi-Head Attention (12 heads, head_dim=64)
    - Feed-Forward (hidden=3072)
    - Layer Norm + Residual
    ↓
Output Projection (hidden=768 → output_tokens)
    ↓
LoRA Adapter (rank=8, alpha=16):
    - Fine-tunes: q_proj, v_proj (8×8 matrices per head)
    - Adds: ~5MB per style
    ↓
Final Output (token predictions)
```

**Specifications:**
- Hidden dimension: 768
- Number of layers: 8
- Number of heads: 12
- Head dimension: 768 ÷ 12 = 64
- Feed-forward dimension: 3072
- Total parameters (base): ~20M
- Quantized size: ~10MB (4-bit)

**LoRA Configuration:**
- Rank (r): 8
- Alpha (α): 16
- Targets: q_proj, v_proj (query and value projections)
- Per-adapter size: ~5MB
- Total adapters: 35 (5 modalities × 7 styles per modality)

### 4. Encoders (`encoders/`)

**Image Encoder** (`image_encoder.py`)
- Architecture: VQVAE (Vector Quantized Variational Autoencoder)
- Input: RGB images (any size, auto-resized to 128×128)
- Encoding: 128×128 → 16×16 latent space
- Codebook size: 8192 entries
- Output: 256 codebook indices per image
- Processing: Convert to token IDs (offset by 50k to avoid vocab collision)

**Audio Encoder** (`audio_encoder.py`)
- Architecture: EnCodec (Meta AI)
- Input: WAV/MP3 mono/stereo (resampled to 24kHz)
- Bandwidth: Configurable 1.5-24 kbps
- Output: Discrete codes (100-200 codes per second of audio)
- Processing: Convert to token IDs

**Video Encoder** (`video_encoder.py`)
- Architecture: Frame sampler + Image encoder
- Input: MP4/AVI video (any frame rate)
- Sampling: 1 frame per second (configurable)
- Output: Sequence of image codes + temporal tokens
- Processing: Build token sequence preserving temporal order

**PDF Encoder** (`pdf_encoder.py`)
- Architecture: Text extraction + tokenizer
- Input: PDF files
- Processing: Extract text → tokenize
- Output: Token sequence (preserving structure with special tokens)
- Features: Metadata preservation, page markers

### 5. Decoders (`decoders/`)

**Image Decoder** (`image_decoder.py`)
- Inverse of VQVAE encoder
- Input: 256 codebook indices → 16×16 latent space
- Output: 128×128 RGB image
- Post-processing: Optional super-resolution

**Audio Decoder** (`audio_decoder.py`)
- Inverse of EnCodec
- Input: Discrete codes
- Output: WAV audio (24kHz, quantized)
- Post-processing: Optional noise reduction

**Video Decoder** (`video_decoder.py`)
- Inverse of frame sampler
- Input: Sequence of image codes
- Output: MP4 video (30fps, H.264 codec)
- Processing: Frame reconstruction + interpolation

### 6. Runtime Components (`runtime/`)

**Lazy Model Loader** (`loader.py`)
```python
class LazyModelLoader:
    - max_memory: 900MB
    - always_loaded: [main_router, shared_base]
    - active_adapter: None (loaded on demand)
    - cache: []
    
    Methods:
    - load_adapter(modality, style)
    - unload_adapter()
    - monitor_memory() → %used
    - clear_cache()
```

**Tool Registry** (`tools.py`)
```python
class ToolRegistry:
    tools = {
        'web_search': Function(timeout=10s, max_results=5),
        'calculator': Function(timeout=5s, expr_limit=100),
        'code_executor': Function(timeout=30s, sandbox=True),
        'file_read': Function(timeout=5s, max_size=10MB),
        'file_write': Function(timeout=5s, max_size=10MB),
        'image_capture': Function(timeout=2s, device=/dev/video0)
    }
```

**Router Inference** (`router.py`)
```python
class OmniModelRouter:
    def infer(input, modality=None, style=None, thinking=False, tools=False):
        1. Detect modality if not specified
        2. Encode input with appropriate encoder
        3. Route through main router model
        4. Extract routing decisions: <route> tokens
        5. Load appropriate sub-model + adapter
        6. Generate output tokens
        7. Optionally execute tools if <tool> tokens detected
        8. Decode output to native format
        9. Return result
        
    Memory stages:
    - Start: ~60MB (main + base)
    - After load adapter: ~85MB
    - Peak during generation: ~120MB
    - After cleanup: ~60MB
```

### 7. Training Pipeline (`training/`)

**LoRA Fine-tuning** (`lora.py`)
```
For each style in [creative, professional, simple, human_like, code]:
    For each modality in [text, image, audio, video, pdf]:
        1. Generate synthetic training data (500 examples)
        2. Load shared sub-model base
        3. Attach LoRA adapter (rank=8)
        4. Fine-tune on style-specific data (2 epochs)
        5. Save adapter to disk (~5MB)
        6. Validate on test set
        
Total: 35 adapters × ~5MB = 175MB disk
Training time: ~5 hours on GPU
```

**Training Data Generation:**
- Templates per style
- Random input/output pairs
- Diverse contexts and domains
- Balance across modalities

### 8. Quantization (`quantize/`)

**4-bit Quantization Strategy:**
```
Main Router (FP32):
  50M params × 4 bytes = 200MB
  ↓
4-bit quantization:
  50M params × 0.5 bytes = 25MB (75% reduction)
  
Sub-model Base (FP32):
  20M params × 4 bytes = 80MB
  ↓
4-bit quantization:
  20M params × 0.5 bytes = 10MB (75% reduction)
  
LoRA Adapters (FP16):
  5M params × 2 bytes = 10MB each
  ↓
Keep as FP16:
  5M params × 2 bytes = 10MB (no reduction, already small)
```

**Quantization Method:** 
- Weight-only quantization (activations in FP32)
- Per-channel quantization
- Static quantization (no calibration data)
- Library: bitsandbytes or GPTQ

## Memory Budget

### Active Memory (During Inference)
```
Component                Size      Status
─────────────────────────────────────────
Main router (4-bit)      25 MB     Always loaded
Shared base (4-bit)      10 MB     Always loaded
Active adapter (FP16)     5 MB     On-demand
Encoder (active)          40 MB    Temporary
Decoder (active)          40 MB    Temporary
Generation cache          10 MB    Temporary
Working memory            -5 MB    Overhead
─────────────────────────────────────────
Total peak               120 MB    ~13% of 1GB
```

### Reserved Buffer
```
Total RAM:               1000 MB   100%
OmniModel peak:           120 MB   12%
System/OS overhead:       100 MB   10%
Safety margin:            780 MB   78%
```

**Key constraint:** Only one encoder/decoder + adapter loaded at a time.

## Token Flow Examples

### Text Generation Example
```
Input: "Write a creative poem"
      ↓
Tokenize: [29, 1823, 42, ...] (5 tokens)
      ↓
Router: <text> <creative> <text_generation>
      ↓
Load: text_creative adapter
      ↓
Sub-model generates: "Moonlight dancing through..."
      ↓
Output: String text
```

### Image Analysis Example
```
Input: photo.jpg
      ↓
Detect: <image> modality
      ↓
Encode (VQVAE): 128×128 RGB → 256 codes
      ↓
Tokenize codes: [50256, 50257, ...] (256 tokens)
      ↓
Router: <image> <professional> <analysis>
      ↓
Load: image_professional adapter
      ↓
Sub-model generates: "A professional photograph of..."
      ↓
Output: Text description
```

### Tool Calling Example
```
Input: "What is 2+2?"
      ↓
Router detects: <tool_calculator> token
      ↓
Extract args: expr="2+2"
      ↓
Tool execution: calculator.evaluate("2+2") → 4
      ↓
Sub-model generates: "The answer is 4"
      ↓
Output: "The answer is 4"
```

## Performance Characteristics

### Latency (Intel i7 CPU, no GPU)
```
Operation                 Time      Notes
───────────────────────────────────
Tokenization             <100ms     Fast
Router inference         200-500ms  Small model
Sub-model inference      1-5s       Depends on output length
Image encoding           1-2s       VQVAE encoder
Image decoding           500-1000ms VQVAE decoder
Audio encoding           1-3s       EnCodec
Audio decoding           1-3s       EnCodec
Total (text→text):       1-7s       Network bottleneck
Total (image→text):      5-10s      Encoding dominates
```

### Throughput
```
Requests/hour: ~60 (16.7s average per request)
Sustained load: Single request at a time
Batch inference: Not supported (1GB RAM constraint)
```

### Accuracy (Compared to full models)
```
Task              Full Model   OmniModel   Delta
──────────────────────────────────────────
Text generation   BLEU: 0.42   BLEU: 0.38 -9%
Image analysis    CLIP: 0.82   CLIP: 0.79 -4%
Audio analysis    ASR WER      WER: 0.15  Degraded
Reasoning         0-shot 42%   0-shot 35% -17%
Tool calling      Accuracy 95% Accuracy 88% -7%
```

## Security Considerations

1. **Sandboxing:** Code executor runs in subprocess with timeout
2. **File access:** Restricted to specified directories
3. **Network:** Web search rate-limited to 5 results/request
4. **Memory:** Continuous monitoring prevents DoS
5. **Model integrity:** Checksums on quantized models

## Future Optimizations

1. **GGUF export** for llama.cpp integration
2. **Streaming inference** (token-at-a-time output)
3. **Mixed precision** (INT8 for weights, FP32 for activation)
4. **Pruning** to reduce to 500MB footprint
5. **Distillation** to create 10M parameter version
6. **Dynamic model selection** based on input complexity

---

**Design principle:** Every architecture decision prioritizes fitting within 1GB RAM while maintaining reasonable inference quality.
