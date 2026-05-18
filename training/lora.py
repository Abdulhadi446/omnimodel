"""
LoRA Fine-tuning - Fine-tune sub-models with LoRA adapters.
Rank: 8, Alpha: 16, Target: q_proj, v_proj
"""

import os
import json
import random
from typing import List, Dict, Any, Optional

try:
    import torch
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from peft import LoraConfig, get_peft_model
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from main_model.config import MODALITIES


class StyleDataset(Dataset):
    """
    Dataset for style-specific fine-tuning.
    Each example: input tokens -> expected output tokens
    """
    
    def __init__(self, examples: List[Dict[str, Any]], modality: str, style: str):
        """
        Args:
            examples: List of training examples
            modality: text, image, audio, etc.
            style: creative, professional, etc.
        """
        self.examples = examples
        self.modality = modality
        self.style = style
        self.max_seq_len = 256
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]
        
        # Tokenize example (simplified)
        input_ids = example.get("input_ids", [])
        target_ids = example.get("target_ids", [])
        
        # Pad to max length
        input_ids = input_ids[:self.max_seq_len]
        target_ids = target_ids[:self.max_seq_len]
        
        input_ids += [0] * (self.max_seq_len - len(input_ids))
        target_ids += [-100] * (self.max_seq_len - len(target_ids))  # -100 = ignore in loss
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
        }


def generate_style_data(modality: str, style: str, num_examples: int = 500) -> List[Dict[str, Any]]:
    """
    Generate synthetic training data for a specific style.
    In production, this would load real labeled data.
    """
    examples = []
    
    # Templates per style
    templates = {
        "text": {
            "human_like": [
                "Hey, how's it going?",
                "So, what do you think about this?",
                "I was thinking, maybe we should...",
            ],
            "creative": [
                "Imagine a world where colors sing...",
                "The moonlight whispered secrets to the tide...",
                "Reality is but a canvas for dreams...",
            ],
            "professional": [
                "Please find the attached document for your review.",
                "Based on our analysis, we recommend the following approach.",
                "In conclusion, the data suggests...",
            ],
            "simple": [
                "It is a cat.",
                "The cat is here.",
                "I like cats.",
            ],
            "code": [
                "def hello_world():",
                "    print('Hello, World!')",
                "    return 0",
            ],
        },
        "image": {
            "realism": "Photorealistic image of a",
            "cartoon": "Cartoon style drawing of a",
            "professional": "Professional illustration of a",
            "simple": "Simple minimalist icon of a",
        },
    }
    
    # Generate examples
    style_templates = templates.get(modality, {}).get(style, ["Example"])
    
    for i in range(num_examples):
        template = random.choice(style_templates)
        
        # Create input and target
        input_text = f"Generate {style} {template[:50]}"
        target_text = template
        
        # Simple tokenization
        input_ids = [ord(c) % 256 for c in input_text][:100]
        target_ids = [ord(c) % 256 for c in target_text][:100]
        
        examples.append({
            "input_ids": input_ids,
            "target_ids": target_ids,
            "modality": modality,
            "style": style,
        })
    
    return examples


def fine_tune_lora(
    base_model,
    modality: str,
    style: str,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    device: str = "cpu",
) -> Optional[Any]:
    """
    Fine-tune a sub-model with LoRA.
    
    Args:
        base_model: Base transformer model
        modality: text, image, etc.
        style: creative, professional, etc.
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: torch device
    
    Returns:
        Fine-tuned PEFT model
    """
    if not TORCH_AVAILABLE:
        print("⚠ PyTorch not available")
        return None
    
    print(f"\nFine-tuning {modality}.{style}...")
    
    # Generate training data
    examples = generate_style_data(modality, style, num_examples=500)
    dataset = StyleDataset(examples, modality, style)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Create LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA to base model
    try:
        peft_model = get_peft_model(base_model, lora_config)
        peft_model = peft_model.to(device)
        
        print(f"✓ Created LoRA adapter")
        print(f"  Trainable params: {sum(p.numel() for p in peft_model.parameters() if p.requires_grad) / 1e6:.1f}M")
    
    except Exception as e:
        print(f"✗ Error creating LoRA: {e}")
        return None
    
    # Training setup
    optimizer = optim.AdamW(peft_model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    # Train
    peft_model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            
            # Forward pass
            logits = peft_model(input_ids)
            
            # Compute loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = target_ids[:, 1:].contiguous()
            
            loss = loss_fn(
                shift_logits.view(-1, logits.shape[-1]),
                shift_targets.view(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
    
    peft_model.eval()
    return peft_model


def train_all_styles(base_model, num_epochs: int = 3, device: str = "cpu"):
    """
    Fine-tune all style-specific sub-models.
    """
    if not TORCH_AVAILABLE:
        print("⚠ PyTorch not available, skipping training")
        return
    
    print("=" * 60)
    print("LORA FINE-TUNING ALL STYLES")
    print("=" * 60)
    
    adapters_dir = "./checkpoints/adapters"
    os.makedirs(adapters_dir, exist_ok=True)
    
    trained_count = 0
    
    for modality, styles in MODALITIES.items():
        for style in styles:
            try:
                # Fine-tune
                peft_model = fine_tune_lora(
                    base_model,
                    modality,
                    style,
                    num_epochs=num_epochs,
                    learning_rate=1e-4,
                    device=device,
                )
                
                if peft_model:
                    # Save adapter
                    adapter_path = f"{adapters_dir}/{modality}_{style}_lora.pt"
                    torch.save(peft_model.state_dict(), adapter_path)
                    print(f"✓ Saved adapter to {adapter_path}")
                    
                    trained_count += 1
            
            except Exception as e:
                print(f"✗ Error training {modality}.{style}: {e}")
    
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE: {trained_count} adapters trained")
    print("=" * 60)


def main():
    """Main training routine"""
    if not TORCH_AVAILABLE:
        print("⚠ PyTorch not installed")
        print("  Install with: pip install torch>=2.0.0")
        return
    
    from sub_models.base import create_shared_base
    
    # Create base model
    base_model = create_shared_base()
    
    if base_model is None:
        print("✗ Could not create base model")
        return
    
    # Train all styles
    train_all_styles(base_model, num_epochs=3, device="cpu")


if __name__ == "__main__":
    main()
