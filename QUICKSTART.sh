#!/bin/bash
# QUICKSTART.sh - Quick setup, test, and train OmniModel

set -e

echo "=========================================="
echo "OmniModel - Quick Start"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version found"

# Install dependencies directly
echo "Installing dependencies..."
python3 -m pip install -q torch transformers peft psutil numpy scipy einops safetensors --no-warn-script-location

# Optional: Install for full functionality
echo "Installing optional dependencies..."
python3 -m pip install -q encodec pypdf2 pillow requests --no-warn-script-location 2>/dev/null || true

echo "✓ Dependencies installed"

# Run tests
echo ""
echo "Running tests..."
python3 tests/test_all.py

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Test inference:"
echo "    python3 cli.py --input \"hello world\""
echo ""
echo "2. Start training (CPU or GPU):"
echo "    python3 train.py"
echo ""
echo "3. Training examples:"
echo "    python3 cli.py --input \"write a poem\" --style creative"
echo "    python3 cli.py --input \"what is 2+2\" --tools"
echo "    python3 cli.py --input \"test\" --output result.txt"
echo ""
echo "4. Monitor memory usage:"
echo "    watch -n 1 free -h"
echo ""
echo "=========================================="
