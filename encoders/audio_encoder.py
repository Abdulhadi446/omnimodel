"""
Audio Encoder - Wraps Meta's EnCodec for efficient audio tokenization.
"""

from typing import List, Optional

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class AudioEncoder:
    """
    Wraps EnCodec for audio tokenization.
    24kHz, lowest bandwidth setting.
    """
    
    def __init__(self, bandwidth: str = "1.5"):
        """
        Args:
            bandwidth: "1.5", "3", "6", "12", or "24" kbps
        """
        self.bandwidth = bandwidth
        
        if TORCH_AVAILABLE:
            try:
                # Import encodec
                import encodec
                from encodec.models import CompressionModel
                
                # Load EnCodec model
                self.model = CompressionModel.encodec_model_24khz()
                self.model.eval()
                self.sample_rate = 24000
                
                print(f"✓ Created AudioEncoder (EnCodec 24kHz, {bandwidth} kbps)")
            except ImportError:
                print("⚠ EnCodec not available, install with: pip install encodec")
                self.model = None
        else:
            print("⚠ PyTorch not available for audio encoder")
            self.model = None
    
    def encode(self, audio_path: str) -> List[int]:
        """
        Encode an audio file to tokens.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            List of encoded tokens
        """
        if self.model is None:
            print("⚠ EnCodec model not available")
            return []
        
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample to 24kHz if needed
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)
            
            # Encode with EnCodec
            with torch.no_grad():
                encoded_frames = self.model.encode(waveform.unsqueeze(0))
            
            # Extract codes (tokens)
            codes = encoded_frames[0][0]  # [batch, n_codes, timesteps]
            token_ids = codes.flatten().tolist()
            
            return token_ids
        
        except Exception as e:
            print(f"✗ Error encoding audio: {e}")
            return []
    
    def encode_bytes(self, audio_bytes: bytes, format: str = "wav") -> List[int]:
        """
        Encode audio from bytes.
        """
        if self.model is None:
            return []
        
        try:
            import io
            
            # Load from bytes
            audio, sr = torchaudio.load(io.BytesIO(audio_bytes), format=format)
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                audio = resampler(audio)
            
            # Encode
            with torch.no_grad():
                encoded_frames = self.model.encode(audio.unsqueeze(0))
            
            codes = encoded_frames[0][0]
            return codes.flatten().tolist()
        
        except Exception as e:
            print(f"✗ Error encoding audio bytes: {e}")
            return []


def create_audio_encoder(bandwidth: str = "1.5") -> Optional[AudioEncoder]:
    """Create an audio encoder"""
    return AudioEncoder(bandwidth=bandwidth)


if __name__ == "__main__":
    encoder = create_audio_encoder()
    if encoder:
        print("✓ Audio encoder created")
