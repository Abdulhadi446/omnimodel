"""
Video Encoder - Samples frames and encodes them using image encoder.
Max 8 frames (2048 tokens total).
"""

from typing import List, Optional
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class VideoEncoder:
    """
    Encodes video by sampling frames and encoding each with VQVAE.
    """
    
    def __init__(self, image_encoder=None, fps: int = 1, max_frames: int = 8):
        """
        Args:
            image_encoder: ImageEncoder instance
            fps: Frames per second to sample
            max_frames: Maximum frames to encode
        """
        self.image_encoder = image_encoder
        self.fps = fps
        self.max_frames = max_frames
        
        print(f"✓ Created VideoEncoder ({fps} fps, max {max_frames} frames)")
    
    def encode(self, video_path: str) -> List[List[int]]:
        """
        Encode a video file to frame tokens.
        
        Args:
            video_path: Path to video file
        
        Returns:
            List of frame token lists
        """
        if not CV2_AVAILABLE:
            print("⚠ OpenCV not available, install with: pip install opencv-python")
            return []
        
        if self.image_encoder is None:
            print("⚠ Image encoder not provided")
            return []
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"✗ Cannot open video: {video_path}")
                return []
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps / self.fps)
            
            frames = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample at desired fps
                if frame_count % frame_interval == 0 and len(frames) < self.max_frames:
                    # Resize to 128x128
                    frame = cv2.resize(frame, (128, 128))
                    frames.append(frame)
                
                frame_count += 1
            
            cap.release()
            
            # Encode each frame
            frame_tokens = []
            for frame in frames:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Save to temp file and encode (simplified)
                # In production, pass frame directly to encoder
                try:
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                        frame_path = f.name
                    
                    # Save frame
                    cv2.imwrite(frame_path, frame_rgb)
                    
                    # Encode frame
                    tokens = self.image_encoder.encode_image_file(frame_path)
                    frame_tokens.append(tokens)
                    
                    import os
                    os.unlink(frame_path)
                
                except Exception as e:
                    print(f"✗ Error encoding frame: {e}")
            
            print(f"✓ Encoded {len(frame_tokens)} frames from video")
            return frame_tokens
        
        except Exception as e:
            print(f"✗ Error processing video: {e}")
            return []


def create_video_encoder(image_encoder=None) -> Optional[VideoEncoder]:
    """Create a video encoder"""
    return VideoEncoder(image_encoder=image_encoder)


if __name__ == "__main__":
    encoder = create_video_encoder()
    if encoder:
        print("✓ Video encoder created")
