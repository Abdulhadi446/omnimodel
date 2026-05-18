#!/bin/bash
# QUICKSTART.sh - Quick setup and test of OmniModel

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
echo "Quick start examples:"
echo ""
echo "  Text generation:"
echo "    python cli.py --input \"hello world\""
echo ""
echo "  With style:"
echo "    python cli.py --input \"write a poem\" --style creative"
echo ""
echo "  With tools:"
echo "    python cli.py --input \"what is 2+2\" --tools"
echo ""
echo "  With output file:"
echo "    python cli.py --input \"test\" --output result.txt"
echo ""
echo "  Check memory usage:"
echo "    watch -n 1 free -h"
echo ""
echo "=========================================="
