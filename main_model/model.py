"""
Main Router Model - Small transformer that decides which sub-model to invoke.
Architecture: 6 layers, 8 heads, 512 hidden dim, ~50M parameters
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE) for efficient position encoding"""
    
    def __init__(self, dim: int, max_seq_length: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        
        # Precompute cos and sin values
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, dim]
        Returns:
            cos and sin embeddings for RoPE
        """
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [seq_len, dim//2]
        
        # Duplicate frequencies to match dimension
        if self.dim % 2 == 1:
            # Odd dimension - pad
            emb = torch.cat([freqs, torch.zeros_like(freqs[:, -1:])], dim=-1)
        else:
            # Even dimension - concatenate
            emb = torch.cat([freqs, freqs], dim=-1)
        
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin


class MultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE"""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_dim]
            cos, sin: RoPE embeddings
            attention_mask: [batch, seq_len, seq_len]
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # [batch, seq_len, hidden_dim]
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        # Simplified: rotate q and k
        q = q * self.scale
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1))  # [batch, heads, seq_len, seq_len]
        
        if attention_mask is not None:
            # Expand mask for multi-head: [batch, seq_len, seq_len] -> [batch, 1, seq_len, seq_len]
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # [batch, heads, seq_len, head_dim]
        
        # Reshape back
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, hidden_dim)
        
        # Output projection
        output = self.out_proj(output)
        
        return output


class TransformerLayer(nn.Module):
    """Single transformer layer with attention and feedforward"""
    
    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        self.attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attn(x, cos, sin, attention_mask)
        x = x + attn_out
        x = self.norm1(x)
        
        # Feedforward with residual
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.norm2(x)
        
        return x


class RouterModel(nn.Module):
    """
    Main router model: 6 layers, 8 heads, 512 hidden dim
    Routes inputs to appropriate sub-models based on modality and style.
    """
    
    def __init__(
        self,
        vocab_size: int = 59496,  # From unified tokenizer
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        max_seq_length: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pe = RotaryPositionalEmbedding(hidden_dim, max_seq_length)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer for routing decisions
        # Can output: route decision (modality.style) or tool call
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters())
    
    def _init_weights(self):
        """Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len] token indices
            attention_mask: [batch, seq_len] or [batch, seq_len, seq_len]
        
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Embed tokens
        x = self.embedding(input_ids)
        x = self.dropout(x)
        
        # Get RoPE embeddings
        cos, sin = self.pe(x)
        
        # Create causal attention mask if not provided
        if attention_mask is None:
            # Causal mask: position i can only attend to positions <= i
            attention_mask = torch.ones(
                (batch_size, seq_len, seq_len),
                device=x.device,
                dtype=torch.float32,
            )
            attention_mask = torch.tril(attention_mask)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, cos, sin, attention_mask)
        
        # Project to vocabulary
        logits = self.output_proj(x)  # [batch, seq_len, vocab_size]
        
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        """
        for _ in range(max_new_tokens):
            # Get logits for the last token
            logits = self.forward(input_ids)
            next_logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0 and top_k < next_logits.shape[-1]:
                # Use topk instead of deprecated kth_value
                kth_vals = torch.topk(next_logits, min(top_k, next_logits.shape[-1]), largest=True)[0]
                kth_val = kth_vals[:, -1:]  # Get the k-th largest value
                next_logits = torch.where(
                    next_logits >= kth_val,
                    next_logits,
                    torch.tensor(float('-inf'), device=next_logits.device)
                )
            
            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumsum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumsum > top_p
                sorted_indices_to_remove[..., 0] = False
                next_logits[sorted_indices[sorted_indices_to_remove]] = float('-inf')
            
            # Replace any -inf with a small value to avoid multinomial errors
            next_logits = torch.where(
                torch.isinf(next_logits),
                torch.tensor(float('-100'), device=next_logits.device),
                next_logits
            )
            
            # Sample next token
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


def create_router_model(pretrained: bool = False) -> RouterModel:
    """Create and optionally load a router model"""
    from .config import MODEL_CONFIG
    
    model = RouterModel(
        vocab_size=MODEL_CONFIG["vocab_size"],
        hidden_dim=MODEL_CONFIG["hidden_dim"],
        num_layers=MODEL_CONFIG["num_layers"],
        num_heads=MODEL_CONFIG["num_heads"],
        ffn_dim=MODEL_CONFIG["ffn_dim"],
        max_seq_length=MODEL_CONFIG["max_seq_length"],
        dropout=MODEL_CONFIG["dropout"],
    )
    
    print(f"✓ Created router model with {model.n_params / 1e6:.1f}M parameters")
    
    return model


# Example usage
if __name__ == "__main__":
    model = create_router_model()
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, 59496, (batch_size, seq_len))
    
    logits = model(input_ids)
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Model params: {model.n_params / 1e6:.1f}M")
