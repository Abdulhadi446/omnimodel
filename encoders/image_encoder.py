"""
Image Encoder - VQVAE that compresses 128x128 RGB images to tokens.
Encodes: 128x128 image -> 16x16 grid of codes (256 tokens)
"""

import numpy as np
from typing import List, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class VQVAEEncoder(nn.Module if TORCH_AVAILABLE else object):
    """
    Tiny VQVAE encoder that compresses images to tokens.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 128,
        num_res_blocks: int = 2,
        num_codebook: int = 8192,  # 8192 possible tokens
    ):
        if not TORCH_AVAILABLE:
            print("⚠ PyTorch not available for VQVAE")
            return
        
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_codebook = num_codebook
        
        # Encoder: 128x128 -> 16x16
        # Each stride-2 layer: 128 -> 64 -> 32 -> 16
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Codebook for vector quantization
        self.codebook = nn.Embedding(num_codebook, hidden_dim * 2)
        
        print(f"✓ Created VQVAEEncoder ({hidden_dim}x2 codebook, {num_codebook} codes)")
    
    def forward(self, x: torch.Tensor) -> Tuple[List[int], torch.Tensor]:
        """
        Encode image to tokens.
        
        Args:
            x: [batch, channels, height, width] image tensor
        
        Returns:
            token_ids: List of codebook indices (256 tokens)
            encoded: Encoded feature map
        """
        if not TORCH_AVAILABLE:
            return [], None
        
        # Encode
        encoded = self.encoder(x)  # [batch, hidden_dim*2, 16, 16]
        
        # Flatten spatial dims for quantization
        batch_size, channels, h, w = encoded.shape
        encoded_flat = encoded.view(batch_size, channels, -1)  # [batch, channels, 256]
        encoded_flat = encoded_flat.permute(0, 2, 1)  # [batch, 256, channels]
        
        # Vector quantization: find nearest codebook entries
        distances = torch.cdist(
            encoded_flat.view(-1, channels),
            self.codebook.weight
        )  # [batch*256, num_codebook]
        
        token_ids = torch.argmin(distances, dim=1)  # [batch*256]
        token_ids = token_ids.view(batch_size, -1)  # [batch, 256]
        
        return token_ids, encoded
    
    def encode_image_file(self, image_path: str) -> List[int]:
        """
        Encode an image file to tokens.
        """
        if not TORCH_AVAILABLE:
            print("⚠ PyTorch not available")
            return []
        
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img = img.resize((128, 128))
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)  # [1, 3, 128, 128]
            
            with torch.no_grad():
                token_ids, _ = self.forward(img)
            
            return token_ids[0].tolist()
        except Exception as e:
            print(f"✗ Error encoding image: {e}")
            return []


def create_image_encoder() -> Optional[VQVAEEncoder]:
    """Create an image encoder"""
    if not TORCH_AVAILABLE:
        print("⚠ PyTorch not available for image encoder")
        return None
    
    return VQVAEEncoder(
        in_channels=3,
        hidden_dim=128,
        num_res_blocks=2,
        num_codebook=8192,
    )


if __name__ == "__main__":
    encoder = create_image_encoder()
    if encoder:
        print("✓ Image encoder created successfully")
