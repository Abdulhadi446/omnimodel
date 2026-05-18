#!/usr/bin/env python3
"""
OmniModel Training Runner - Fine-tune all LoRA adapters
"""

import sys
import os

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    print(f"✓ PyTorch {torch.__version__} available")
except ImportError:
    print("✗ PyTorch not installed. Install with: pip install torch>=2.0.0")
    sys.exit(1)

from training.lora import train_all_styles
from sub_models.base import create_shared_base

def main():
    """Main training routine"""
    print("=" * 60)
    print("OMNIMODEL TRAINING")
    print("=" * 60)
    
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n✓ Using device: {device}")
    
    if device == "cpu":
        print("⚠ Training on CPU - this will be slow")
        print("  For faster training, use a GPU or cloud service")
    
    # Create base model
    print("\nCreating base model...")
    base_model = create_shared_base()
    
    if base_model is None:
        print("✗ Could not create base model")
        return 1
    
    base_model = base_model.to(device)
    
    # Train all styles
    print(f"\n✓ Base model ready ({sum(p.numel() for p in base_model.parameters()) / 1e6:.1f}M params)")
    
    try:
        train_all_styles(base_model, num_epochs=1, device=device)
        print("\n✓ Training complete!")
        return 0
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
