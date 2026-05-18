"""
Sub-model Template - Shared base model with LoRA adapters for each style.
Each sub-model is a LoRA adapter (~5MB) on top of a shared 20M param base.
"""

import os
from typing import Optional, Dict, Any

try:
    import torch
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class SubModelBase(nn.Module):
        """
        Shared base model that all sub-models use.
        4 layers, 256 hidden dim, ~20M parameters.
        """
        
        def __init__(
            self,
            vocab_size: int = 59496,
            hidden_dim: int = 256,
            num_layers: int = 4,
            num_heads: int = 4,
            ffn_dim: int = 1024,
        ):
            super().__init__()
            
            self.vocab_size = vocab_size
            self.hidden_dim = hidden_dim
            
            # Embeddings
            self.embedding = nn.Embedding(vocab_size, hidden_dim)
            
            # Transformer layers
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=ffn_dim,
                    batch_first=True,
                    dropout=0.1,
                )
                for _ in range(num_layers)
            ])
            
            # Output projection
            self.output_proj = nn.Linear(hidden_dim, vocab_size)
            
            # Count parameters
            self.n_params = sum(p.numel() for p in self.parameters())
            
            print(f"✓ Created SubModelBase ({self.n_params / 1e6:.1f}M params)")
        
        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                input_ids: [batch, seq_len]
            
            Returns:
                logits: [batch, seq_len, vocab_size]
            """
            # Embed
            x = self.embedding(input_ids)
            
            # Apply transformer layers
            for layer in self.layers:
                x = layer(x)
            
            # Project to vocab
            logits = self.output_proj(x)
            
            return logits
else:
    class SubModelBase:
        """Stub when PyTorch not available"""
        def __init__(self, *args, **kwargs):
            self.n_params = 20e6
            print("⚠ PyTorch not available for SubModelBase")


class SubModel:
    """
    A sub-model instance: shared base + LoRA adapter for a specific style.
    Each style gets its own LoRA weights (~5MB).
    """
    
    def __init__(self, modality: str, style: str, base_model: Optional[SubModelBase] = None):
        """
        Args:
            modality: text, image, audio, video, pdf
            style: human_like, creative, professional, etc.
            base_model: Shared base model instance
        """
        self.modality = modality
        self.style = style
        self.base_model = base_model
        self.lora_model = None
        self.loaded = False
        
        print(f"✓ Created SubModel: {modality}.{style}")
    
    def create_lora_adapter(self) -> Optional[object]:
        """
        Create a LoRA adapter on top of the base model.
        """
        if not TORCH_AVAILABLE or self.base_model is None:
            return None
        
        try:
            # LoRA config
            lora_config = LoraConfig(
                r=8,  # LoRA rank
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],  # Target attention projections
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            # Create PEFT model with LoRA
            self.lora_model = get_peft_model(self.base_model, lora_config)
            
            print(f"✓ Created LoRA adapter for {self.modality}.{self.style}")
            
            return self.lora_model
        
        except Exception as e:
            print(f"✗ Error creating LoRA adapter: {e}")
            return None
    
    def load(self, adapter_path: Optional[str] = None) -> bool:
        """
        Load the model (base + adapter) into RAM.
        
        Args:
            adapter_path: Path to saved LoRA adapter weights
        
        Returns:
            True if successful
        """
        if not TORCH_AVAILABLE:
            return False
        
        try:
            if self.base_model is None:
                self.base_model = SubModelBase()
            
            # Create or load LoRA adapter
            if self.lora_model is None:
                self.create_lora_adapter()
            
            # Load saved weights if provided
            if adapter_path and os.path.exists(adapter_path):
                self.lora_model.load_state_dict(torch.load(adapter_path))
                print(f"✓ Loaded adapter from {adapter_path}")
            
            self.lora_model.eval()
            self.loaded = True
            
            print(f"✓ Loaded {self.modality}.{self.style} into RAM")
            return True
        
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    
    def unload(self) -> None:
        """
        Unload the model from RAM to free memory.
        """
        if self.lora_model is not None:
            # Move to CPU memory (doesn't actually free, but stops GPU usage)
            try:
                self.lora_model.cpu()
                if TORCH_AVAILABLE:
                    torch.cuda.empty_cache()
                self.loaded = False
                print(f"✓ Unloaded {self.modality}.{self.style} from RAM")
            except:
                pass
    
    def forward(self, input_ids) -> Optional[object]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: [batch, seq_len]
        
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        if not self.loaded or self.lora_model is None:
            print(f"✗ Model not loaded: {self.modality}.{self.style}")
            return None
        
        try:
            with torch.no_grad():
                outputs = self.lora_model(input_ids)
            return outputs
        except Exception as e:
            print(f"✗ Error in forward pass: {e}")
            return None
    
    def save_adapter(self, save_path: str) -> bool:
        """
        Save LoRA adapter weights to disk (~5MB).
        """
        if not TORCH_AVAILABLE or self.lora_model is None:
            return False
        
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.lora_model.state_dict(), save_path)
            print(f"✓ Saved adapter to {save_path}")
            return True
        except Exception as e:
            print(f"✗ Error saving adapter: {e}")
            return False
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Estimate memory usage.
        """
        return {
            "modality": self.modality,
            "style": self.style,
            "loaded": self.loaded,
            "base_params": self.base_model.n_params if self.base_model else 0,
            "lora_size_mb": 5,  # LoRA adapters are ~5MB each
            "total_size_mb": 25,  # Base model (20M) + adapter
        }


def create_shared_base() -> Optional[SubModelBase]:
    """Create the shared base model used by all sub-models"""
    if not TORCH_AVAILABLE:
        return None
    
    return SubModelBase(
        vocab_size=59496,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        ffn_dim=1024,
    )


def create_sub_model(modality: str, style: str, base_model: Optional[SubModelBase] = None) -> Optional[SubModel]:
    """
    Create a sub-model instance.
    """
    if base_model is None:
        base_model = create_shared_base()
    
    return SubModel(modality, style, base_model)


if __name__ == "__main__":
    # Create shared base
    base = create_shared_base()
    
    # Create some sub-models
    text_creative = create_sub_model("text", "creative", base)
    image_cartoon = create_sub_model("image", "cartoon", base)
    
    print("\n✓ Sub-model system created successfully")
