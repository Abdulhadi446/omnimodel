# OmniModel - Deployment Checklist

## Pre-Deployment (GPU Machine or Colab)

### Data Preparation
- [ ] Collect or generate training data for each style
- [ ] Validate data quality and diversity
- [ ] Create synthetic data if real data unavailable

### Model Training
- [ ] Train main router model on 1000 synthetic examples
  ```bash
  python main_model/train.py
  ```
- [ ] Verify router outputs correct routing decisions
- [ ] Train LoRA adapters for all styles
  ```bash
  python training/lora.py
  ```
- [ ] Validate adapter quality on test set

### Quantization
- [ ] Quantize main router to 4-bit
- [ ] Quantize shared sub-model base to 4-bit
- [ ] Keep LoRA adapters in FP16 (already tiny, no need to quantize)
- [ ] Verify quantized models produce correct outputs
  ```bash
  python quantize/quantize.py
  ```

### Testing
- [ ] Run full test suite
  ```bash
  python tests/test_all.py
  ```
- [ ] All tests must pass (17 passed, 0 failed)
- [ ] Manual smoke tests on each modality
  - [ ] Text: `python cli.py --input "hello"`
  - [ ] Image: Create test image, run `python cli.py --input test.jpg`
  - [ ] Audio: Create test audio, run `python cli.py --input test.mp3`
  - [ ] PDF: Create test PDF, run `python cli.py --input test.pdf`
  - [ ] Tools: `python cli.py --input "2+2" --tools`

## Packaging for Deployment

### File Selection
- [ ] Copy only quantized models (not training checkpoints)
- [ ] Copy inference-only code (no training scripts)
- [ ] Include minimal requirements.txt for inference

```bash
# Create deployment package
mkdir -p deploy/omnimodel
cp -r models/quantized deploy/omnimodel/
cp -r checkpoints/adapters deploy/omnimodel/
cp omnimodel/encoders deploy/omnimodel/
cp omnimodel/decoders deploy/omnimodel/
cp omnimodel/runtime deploy/omnimodel/
cp omnimodel/tokenizer deploy/omnimodel/
cp omnimodel/cli.py deploy/omnimodel/
cp requirements-inference.txt deploy/requirements.txt
```

### Size Verification
- [ ] Main router quantized: < 30MB
- [ ] Shared sub-base quantized: < 15MB
- [ ] All adapters combined: < 200MB (35 adapters × 5MB)
- [ ] Encoders/decoders: < 100MB
- [ ] Total package: < 400MB
  ```bash
  du -sh deploy/
  ```

## VM Setup (1GB RAM)

### Prerequisites
- [ ] Linux system (Ubuntu 20.04+ recommended)
- [ ] 1GB RAM (can be exactly 1GB)
- [ ] 500MB free disk space
- [ ] Python 3.10+
- [ ] Internet connection (for pip install)

### Installation
```bash
# Extract deployment package
tar xzf omnimodel-deploy.tar.gz
cd omnimodel

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install inference-only dependencies
pip install -r requirements.txt
# Or minimal set:
# pip install torch transformers peft encodec pypdf2 pillow numpy scipy einops safetensors psutil
```

### Checklist
- [ ] Python installed: `python --version` (should be 3.10+)
- [ ] pip available: `pip --version`
- [ ] Dependencies installed without errors
- [ ] All imports work: `python -c "import torch; import transformers; print('OK')"`

## Deployment Verification

### Memory Profiling

**Before first run:**
```bash
free -h  # Check available RAM
ps aux | grep python  # Ensure no other Python processes
```

**During inference:**
```bash
# Terminal 1: Monitor memory
watch -n 1 'free -h; echo "---"; ps aux | grep python | grep -v grep'

# Terminal 2: Run inference
python cli.py --input "test input"
```

**After inference:**
```bash
# Check peak RAM didn't exceed limit
# Expected: Peak usage < 950MB
```

### Functional Tests

- [ ] Text generation works
  ```bash
  python cli.py --input "hello world"
  ```
  Expected: Text output appears within 2 seconds

- [ ] Modality detection works
  ```bash
  python cli.py --input image.jpg
  python cli.py --input audio.mp3
  python cli.py --input document.pdf
  ```
  Expected: Correct modality detected

- [ ] Style forcing works
  ```bash
  python cli.py --input "test" --style creative
  python cli.py --input "test" --style professional
  ```
  Expected: Different styles applied

- [ ] Tool calling works
  ```bash
  python cli.py --input "2+2" --tools
  ```
  Expected: Calculator tool executed, result shown

- [ ] Reasoning works
  ```bash
  python cli.py --input "test" --think
  ```
  Expected: Thinking chain displayed

- [ ] Output saving works
  ```bash
  python cli.py --input "test" --output result.txt
  cat result.txt
  ```
  Expected: Output saved to file

### RAM Constraints Verification

```bash
# Test that system stays under limits
python cli.py --input "hello"  # Should use ~120MB peak
python cli.py --input "test"   # Should unload previous model

# Stress test: Rapid calls
for i in {1..5}; do
  python cli.py --input "test $i"
done
# Monitor that RAM doesn't exceed 950MB during any call
```

### Performance Benchmarks

Run and record timings:

```bash
time python cli.py --input "hello world"  # Text
time python cli.py --input image.jpg      # Image
time python cli.py --input audio.mp3      # Audio
```

Expected times (Intel i7 CPU, no GPU):
- Text: 0.5-2 seconds
- Image: 3-10 seconds
- Audio: 5-20 seconds

## Production Checklist

### Optimization
- [ ] No unnecessary print statements in production code
- [ ] Logging set to ERROR level only
- [ ] Cache cleared between requests if running as service
- [ ] Temporary files cleaned up after processing

### Reliability
- [ ] Error handling for corrupted input files
- [ ] Graceful degradation if model unavailable
- [ ] Timeout protection on tool calls (max 10s)
- [ ] Memory cleanup after OOM situations

### Monitoring
- [ ] Log memory usage to file
- [ ] Track inference times
- [ ] Monitor error rates
- [ ] Alert on RAM usage > 900MB

### Documentation
- [ ] README.md updated with actual performance numbers
- [ ] Known limitations documented
- [ ] Troubleshooting guide created
- [ ] API documentation complete

## Quick Start Commands

Once deployed:

```bash
# Simple test
python cli.py --input "hello"

# Text generation
python cli.py --input "write a poem" --style creative

# Image analysis
python cli.py --input photo.jpg

# Image generation
python cli.py --input "draw a cat" --output-type image

# Tool usage
python cli.py --input "what is 2+2?" --tools

# Full featured
python cli.py --input "hello" --style creative --think --tools
```

## Troubleshooting During Deployment

### Issue: "Out of memory" error
**Solution:**
```bash
# Check running processes
ps aux | grep python

# Kill other Python processes
killall python

# Reduce model size further
python quantize/quantize.py  # Re-quantize with q8_0 instead of q4_0
```

### Issue: "PyTorch not found"
**Solution:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Issue: Import errors
**Solution:**
```bash
# Reinstall all requirements
pip install --upgrade -r requirements.txt
```

### Issue: Slow inference
**Solution:**
```bash
# Normal on CPU - expected 0.5-20 seconds per request
# For faster inference, consider GPU or larger system

# Optimize specific calls
python cli.py --input "short" --style simple  # Simpler = faster
```

### Issue: Memory not freeing after inference
**Solution:**
```bash
# Python garbage collection
import gc
gc.collect()

# Or restart Python process between requests
```

## Success Criteria

You've successfully deployed OmniModel when:

✓ `python cli.py --input "write me a poem"` returns creative text  
✓ `python cli.py --input photo.jpg` returns a description  
✓ `python cli.py --input "draw a cartoon cat" --output-type image` saves an image  
✓ `python cli.py --input recording.mp3` returns a transcription/response  
✓ `python cli.py --input document.pdf` returns a summary  
✓ `python cli.py --input "what is 2+2" --tools` uses calculator tool  
✓ All above work on the 1GB RAM VM  
✓ Peak RAM never exceeds 950MB  

## Support & Debugging

For issues:
1. Check logs: `tail -f omnimodel.log`
2. Run tests: `python tests/test_all.py`
3. Check memory: `free -h`
4. Verify models: `ls -lh models/quantized/`
5. Test individually: `python cli.py --input "test"`

---

**Deployment complete when all checkboxes are checked!**
