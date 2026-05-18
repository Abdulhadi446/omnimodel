"""
Image Decoder - VQVAE decoder that reconstructs images from tokens.
Takes 256 code tokens -> reconstructs 128x128 RGB image.
"""

from typing import List, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class VQVAEDecoder(nn.Module if TORCH_AVAILABLE else object):
    """
    VQVAE decoder that reconstructs images from codebook tokens.
    """
    
    def __init__(
        self,
        num_codebook: int = 8192,
        hidden_dim: int = 128,
        out_channels: int = 3,
    ):
        if not TORCH_AVAILABLE:
            print("⚠ PyTorch not available for VQVAE decoder")
            return
        
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_codebook = num_codebook
        
        # Codebook (same as encoder)
        self.codebook = nn.Embedding(num_codebook, hidden_dim * 2)
        
        # Decoder: 16x16 -> 128x128
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Output in [0, 1]
        )
        
        print(f"✓ Created VQVAEDecoder ({num_codebook} codebook entries)")
    
    def forward(self, token_ids: List[int]) -> Optional[torch.Tensor]:
        """
        Decode tokens to image.
        
        Args:
            token_ids: List of codebook indices (should be 256)
        
        Returns:
            Reconstructed image [1, 3, 128, 128]
        """
        if not TORCH_AVAILABLE:
            return None
        
        try:
            # Convert to tensor
            token_ids = torch.tensor(token_ids, dtype=torch.long)
            
            # Look up in codebook
            codes = self.codebook(token_ids)  # [256, hidden_dim*2]
            
            # Reshape to 16x16 spatial grid
            codes = codes.view(1, 16, 16, -1).permute(0, 3, 1, 2)  # [1, hidden_dim*2, 16, 16]
            
            # Decode
            with torch.no_grad():
                image = self.decoder(codes)  # [1, 3, 128, 128]
            
            return image
        
        except Exception as e:
            print(f"✗ Error decoding image: {e}")
            return None
    
    def decode_to_file(self, token_ids: List[int], output_path: str) -> bool:
        """
        Decode tokens and save image.
        """
        if not TORCH_AVAILABLE:
            return False
        
        try:
            image = self.forward(token_ids)
            if image is None:
                return False
            
            # Convert to PIL Image
            image_np = (image.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            pil_image.save(output_path)
            
            print(f"✓ Saved decoded image to {output_path}")
            return True
        
        except Exception as e:
            print(f"✗ Error saving decoded image: {e}")
            return False


def create_image_decoder() -> Optional[VQVAEDecoder]:
    """Create an image decoder"""
    if not TORCH_AVAILABLE:
        print("⚠ PyTorch not available for image decoder")
        return None
    
    return VQVAEDecoder(
        num_codebook=8192,
        hidden_dim=128,
        out_channels=3,
    )


if __name__ == "__main__":
    decoder = create_image_decoder()
    if decoder:
        print("✓ Image decoder created successfully")
