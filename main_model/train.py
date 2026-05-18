"""
Training script for the main router model.
Trains on synthetic data to learn routing decisions.
"""

import os
import json
import random
from typing import List, Dict, Any

try:
    import torch
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from config import MODEL_CONFIG, TRAINING_CONFIG, MODALITIES
from model import RouterModel


class RouterTrainingDataset(Dataset):
    """
    Synthetic dataset for router training.
    Each example maps an input modality to a route decision.
    """
    
    def __init__(self, examples: List[Dict[str, Any]], tokenizer):
        """
        Args:
            examples: List of training examples
            tokenizer: UnifiedTokenizer instance
        """
        self.examples = examples
        self.tokenizer = tokenizer
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]
        
        # Encode input
        input_tokens = self.tokenizer.encode_with_modality(
            example["input"],
            example["modality"]
        )
        
        # Encode output (route decision)
        output_tokens = self.tokenizer.encode_route_decision(
            example["modality"],
            example["style"]
        )
        
        # Combine as sequence: input + output
        all_tokens = input_tokens + output_tokens
        
        # Pad to fixed length
        max_len = 256
        if len(all_tokens) > max_len:
            all_tokens = all_tokens[:max_len]
        else:
            all_tokens = all_tokens + [0] * (max_len - len(all_tokens))
        
        return {
            "input_ids": torch.tensor(all_tokens, dtype=torch.long),
            "length": min(len(input_tokens + output_tokens), max_len),
        }


def generate_synthetic_data(num_examples: int = 1000) -> List[Dict[str, Any]]:
    """
    Generate synthetic training data for the router.
    Each example: input modality + content -> route decision
    """
    examples = []
    
    # Sample text inputs
    text_samples = [
        "Hello, how are you?",
        "Write me a poem",
        "What is the capital of France?",
        "Draw a cartoon cat",
        "Create a professional document",
        "Generate code for a web server",
        "Analyze this image",
        "Transcribe this audio",
        "Summarize this PDF",
    ]
    
    # Generate examples
    for _ in range(num_examples):
        # Choose random modality and style
        modality_name = random.choice(list(MODALITIES.keys()))
        style = random.choice(MODALITIES[modality_name])
        
        # Create example
        if modality_name == "text":
            content = random.choice(text_samples)
        else:
            content = f"[{modality_name} input]"
        
        example = {
            "input": content,
            "modality": modality_name,
            "style": style,
        }
        
        examples.append(example)
    
    return examples


def train_router(
    model: RouterModel,
    train_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: str = "cpu",
) -> List[float]:
    """
    Train the router model.
    """
    if not TORCH_AVAILABLE:
        print("⚠ PyTorch not available, skipping training")
        return []
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    model = model.to(device)
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            
            # Forward pass
            logits = model(input_ids)  # [batch, seq_len, vocab_size]
            
            # Compute loss (predict next token)
            # Shift targets: predict token at position i+1 from position i
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            
            loss = loss_fn(
                shift_logits.view(-1, logits.shape[-1]),
                shift_labels.view(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
    
    return losses


def save_model(model: RouterModel, path: str):
    """Save model weights"""
    if TORCH_AVAILABLE:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)
        print(f"✓ Saved model to {path}")
    else:
        print(f"⚠ PyTorch not available, skipping save")


def main():
    """Main training routine"""
    print("=" * 60)
    print("ROUTER MODEL TRAINING")
    print("=" * 60)
    
    if not TORCH_AVAILABLE:
        print("⚠ PyTorch not installed")
        print("  Install with: pip install torch>=2.0.0")
        return
    
    # Create model
    model = RouterModel(**MODEL_CONFIG)
    print(f"✓ Created router model ({model.n_params / 1e6:.1f}M params)")
    
    # Generate synthetic training data
    print("\nGenerating synthetic training data...")
    examples = generate_synthetic_data(num_examples=1000)
    print(f"✓ Generated {len(examples)} training examples")
    
    # Create dummy tokenizer reference (would import actual one)
    class DummyTokenizer:
        def encode_with_modality(self, content, modality):
            return [random.randint(0, 59495) for _ in range(10)]
        
        def encode_route_decision(self, modality, style):
            return [random.randint(0, 59495) for _ in range(5)]
    
    tokenizer = DummyTokenizer()
    
    # Create dataset and dataloader
    dataset = RouterTrainingDataset(examples, tokenizer)
    train_loader = DataLoader(
        dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    print(f"✓ Created dataset with {len(dataset)} examples")
    
    # Train
    print("\nTraining...")
    losses = train_router(
        model,
        train_loader,
        num_epochs=TRAINING_CONFIG["num_epochs"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        device=TRAINING_CONFIG["device"],
    )
    
    # Save model
    if TORCH_AVAILABLE:
        save_path = "./checkpoints/router_model.pt"
        save_model(model, save_path)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    return losses


if __name__ == "__main__":
    main()
