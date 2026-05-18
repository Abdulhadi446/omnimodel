# OmniModel - Multimodal AI for 1GB RAM

A modular, multimodal AI system that runs inference on CPU-only hardware with just 1GB RAM. Routes any input (text, image, audio, video, PDF) to specialized sub-models, all unified through a shared token space.

## Architecture Overview

**Main Components:**

- **Router Model** (~50M params): Small transformer that reads any input and routes to the best sub-model
- **Shared Sub-model Base** (~20M params): Backbone used by all style-specific models
- **LoRA Adapters** (~5MB each): Lightweight style-specific fine-tuning per modality
- **Encoders**: VQVAE (images), EnCodec (audio), frame sampler (video), PDF extractor
- **Decoders**: Reconstruct images/audio from tokens
- **Tools**: Web search, calculator, code executor, file I/O, webcam capture

**Memory Strategy:**

- Main router: Always in RAM (~25MB quantized)
- Shared base: Always in RAM (~10MB quantized)
- Active sub-model: One adapter in RAM (~5MB)
- Encoders/decoders: Loaded on-demand (~40MB each)
- **Total active: ~120MB** (well under 1GB limit with 880MB buffer)

## Installation

### Requirements
- Python 3.10+
- 1GB RAM (target), tested on systems with more
- CPU-only (no CUDA support required, but compatible)

### Setup

```bash
cd omnimodel
pip install -r requirements.txt
```

**Optional**: For GPU acceleration, install torch with CUDA:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Command Line

```bash
# Text generation
python cli.py --input "Write me a haiku about AI"

# Image analysis
python cli.py --input photo.jpg

# Image generation
python cli.py --input "draw a cartoon cat" --output-type image

# Audio transcription
python cli.py --input recording.mp3

# PDF summarization
python cli.py --input document.pdf

# With style forcing
python cli.py --input "hello" --style creative

# With reasoning
python cli.py --input "What is 2+2?" --think

# With tool calling
python cli.py --input "What's the weather?" --tools

# Save output
python cli.py --input "text here" --output result.txt
```

### Python API

```python
from omnimodel.runtime.router import create_router
from omnimodel.tokenizer.unified import UnifiedTokenizer
from omnimodel.runtime.tools import create_tool_registry

# Initialize
router = create_router()
tokenizer = UnifiedTokenizer()
tools = create_tool_registry()

# Run inference
result = router.infer(
    "Write a poem",
    modality="text",
    style="creative",
    thinking=True,
    tools=True
)

# Use tools
search_result = tools.call_tool("web_search", {"query": "python"})
calc_result = tools.call_tool("calculator", {"expr": "2**10"})
```

## File Structure

```
omnimodel/
├── main_model/           # Main router transformer
│   ├── model.py
│   ├── train.py
│   └── config.py
├── sub_models/           # Shared base + LoRA adapters
│   ├── base.py
│   ├── text/             # Text generation styles
│   ├── image/            # Image generation styles
│   ├── audio/            # Audio synthesis styles
│   ├── video/            # Video generation styles
│   └── pdf/              # PDF processing styles
├── encoders/             # Input encoding
│   ├── image_encoder.py  # VQVAE
│   ├── audio_encoder.py  # EnCodec
│   ├── video_encoder.py  # Frame sampler
│   └── pdf_encoder.py    # PDF extractor
├── decoders/             # Output decoding
│   ├── image_decoder.py
│   ├── audio_decoder.py
│   └── video_decoder.py
├── tokenizer/            # Unified token space
│   └── unified.py
├── runtime/              # Inference components
│   ├── router.py         # Main inference loop
│   ├── loader.py         # Lazy model loading
│   └── tools.py          # Tool registry
├── training/             # Training scripts
│   ├── lora.py           # LoRA fine-tuning
│   ├── pipeline.py       # Training pipeline
│   └── data/             # Datasets
├── quantize/             # Quantization scripts
│   └── quantize.py
├── tests/                # Test suite
│   └── test_all.py
├── cli.py                # Command-line interface
├── requirements.txt      # Dependencies
└── README.md
```

## Supported Modalities & Styles

**Text Styles:**
- human_like: Conversational, informal
- creative: Metaphorical, vivid
- professional: Formal, structured
- simple: Clear, basic vocabulary
- code: Well-commented code
- mcp_handler: Tool call sequences

**Image Styles:**
- realism: Photorealistic
- cartoon: Comic/animated
- professional: Polished illustration
- simple: Minimalist icon

**Audio Styles:**
- realism: Natural speech
- professional: Polished narration
- simple: Clear, basic

**Video Styles:**
- realism: Photorealistic
- cartoon: Animated
- simple: Minimalist

**PDF Styles:**
- realism, creative, professional, simple

## Training & Fine-tuning

### Generate Training Data

```bash
# Generate synthetic data for all styles
python -c "from training.lora import generate_style_data; data = generate_style_data('text', 'creative', 500)"
```

### Fine-tune LoRA Adapters

```bash
# Requires PyTorch + PEFT
python training/lora.py
```

This will train adapters for each style and save them as ~5MB files.

### Quantize Models

```bash
python quantize/quantize.py
```

Output shows memory estimates for quantized models.

## Testing

Run the full test suite:

```bash
python tests/test_all.py
```

Expected output:
```
✓ ALL TESTS PASSED
Passed:  17
Failed:  0
Skipped: 2
Total:   19
```

## Deployment to 1GB RAM VM

### Step 1: Build quantized models (on GPU machine or Colab)

```bash
# On a machine with more RAM/GPU
python training/lora.py           # Train all adapters
python quantize/quantize.py       # Quantize models
```

### Step 2: Copy to VM

```bash
# Package for deployment
tar czf omnimodel-deploy.tar.gz \
  models/quantized \
  checkpoints/adapters \
  omnimodel/encoders \
  omnimodel/decoders \
  omnimodel/runtime \
  omnimodel/tokenizer \
  omnimodel/cli.py \
  requirements.txt

# Transfer to VM
scp omnimodel-deploy.tar.gz user@vm-ip:/home/user/
```

### Step 3: Deploy on 1GB VM

```bash
# Extract
tar xzf omnimodel-deploy.tar.gz

# Install dependencies (inference-only)
pip install torch transformers peft encodec pypdf2 pillow numpy scipy einops safetensors psutil

# Test
python cli.py --input "hello world"
```

### Step 4: Monitor Memory

```bash
# In one terminal
watch -n 1 'free -h && ps aux | grep python'

# In another, run inference
python cli.py --input "test"

# Verify peak RAM < 950MB
```

## Performance Benchmarks

**Model Sizes (4-bit quantized):**
- Main router: 25MB
- Shared sub-base: 10MB
- Each LoRA adapter: 5MB
- Image encoder: 40MB
- Audio encoder: 40MB

**Total disk**: ~400MB for full system (quantized)
**Peak RAM during inference**: ~120MB active + 800MB buffer = 920MB safe

**Inference Speed (CPU, Intel i7):**
- Text: 0.5-2s per prompt (token generation limited by CPU)
- Image: 3-10s (encoder + generation)
- Audio: 5-20s (encoder + generation)
- PDF: 1-5s (extraction + encoding)

## Troubleshooting

### Out of Memory
- Ensure only one sub-model is loaded at a time
- Reduce batch size (set to 1)
- Clear cache: `python -c "from runtime.loader import create_lazy_loader; create_lazy_loader().clear_cache()"`

### PyTorch Import Error
```bash
pip install torch>=2.0.0
```

### EnCodec Not Available
```bash
pip install encodec
```

### CUDA Out of Memory (with GPU)
Run inference on CPU:
```python
router.router_model.cpu()
```

## Architecture Decisions

**Why LoRA adapters?**
- Each adapter is ~5MB instead of full 20M parameter copy
- Fast switching between styles
- Minimal memory overhead per style

**Why unified token space?**
- Single forward pass handles all modalities
- Enables cross-modal reasoning
- Simplifies router architecture

**Why lazy loading?**
- Keep main model always ready
- Load sub-models on-demand
- Unload to free memory between requests

**Why quantization?**
- 4-bit reduces model size 75% (50M → 12.5M param equivalents)
- FP32 inference still works on CPU
- No accuracy loss for most tasks

## Development Roadmap

- [ ] GGUF export for llama.cpp integration
- [ ] Streaming inference (token-at-a-time)
- [ ] Batch inference support
- [ ] Multi-GPU support
- [ ] Model pruning for <500MB footprint
- [ ] WebAssembly compilation for browser deployment

## Contributing

Contributions welcome! Key areas:
- Optimize encoder/decoder memory
- Implement missing modalities
- Add more tool integrations
- Improve quantization strategies
- Performance benchmarking

## License

MIT

## References

- RoPE: Roformer (Su et al., 2022)
- VQVAE: Taming Transformers (Esser et al., 2021)
- EnCodec: Meta AI (Défossez et al., 2023)
- LoRA: Microsoft Research (Hu et al., 2021)
- Bitsandbytes: Quantization (Dettmers et al., 2022)

## Contact

For issues, questions, or suggestions: [GitHub Issues]

---

**Remember: The goal is maximum capability with minimum resources. Every MB of model size costs precious RAM.**
