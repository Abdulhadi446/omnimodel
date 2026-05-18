"""
Audio Decoder - EnCodec decoder that reconstructs audio from tokens.
"""

from typing import List, Optional

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class AudioDecoder:
    """
    Wraps EnCodec decoder to reconstruct audio from tokens.
    """
    
    def __init__(self, bandwidth: str = "1.5"):
        """
        Args:
            bandwidth: "1.5", "3", "6", "12", or "24" kbps
        """
        self.bandwidth = bandwidth
        self.sample_rate = 24000
        self.model = None
        
        if TORCH_AVAILABLE:
            try:
                import encodec
                from encodec.models import CompressionModel
                
                self.model = CompressionModel.encodec_model_24khz()
                self.model.eval()
                
                print(f"✓ Created AudioDecoder (EnCodec, {bandwidth} kbps)")
            except ImportError:
                print("⚠ EnCodec not available, install with: pip install encodec")
        else:
            print("⚠ PyTorch not available for audio decoder")
    
    def decode(self, token_ids: List[int]) -> Optional[torch.Tensor]:
        """
        Decode tokens to audio waveform.
        
        Args:
            token_ids: List of encoded tokens
        
        Returns:
            Audio waveform [1, num_samples]
        """
        if self.model is None:
            return None
        
        try:
            # Reshape tokens to codes format expected by EnCodec
            codes = torch.tensor(token_ids, dtype=torch.long)
            codes = codes.view(1, -1, 1)  # [1, num_codes, 1]
            
            # Decode with EnCodec
            with torch.no_grad():
                audio = self.model.decode(codes)
            
            return audio
        
        except Exception as e:
            print(f"✗ Error decoding audio: {e}")
            return None
    
    def decode_to_file(self, token_ids: List[int], output_path: str) -> bool:
        """
        Decode tokens and save audio.
        """
        if self.model is None:
            return False
        
        try:
            audio = self.decode(token_ids)
            if audio is None:
                return False
            
            # Save with torchaudio
            torchaudio.save(output_path, audio, self.sample_rate)
            
            print(f"✓ Saved decoded audio to {output_path}")
            return True
        
        except Exception as e:
            print(f"✗ Error saving decoded audio: {e}")
            return False


def create_audio_decoder(bandwidth: str = "1.5") -> Optional[AudioDecoder]:
    """Create an audio decoder"""
    return AudioDecoder(bandwidth=bandwidth)


if __name__ == "__main__":
    decoder = create_audio_decoder()
    if decoder:
        print("✓ Audio decoder created")
