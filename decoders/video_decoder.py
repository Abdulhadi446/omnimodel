"""
Video Decoder - Decodes frames and stitches them into a video.
"""

from typing import List, Optional

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


class VideoDecoder:
    """
    Decodes video frames from token sequences and stitches into video.
    """
    
    def __init__(self, image_decoder=None, fps: int = 1):
        """
        Args:
            image_decoder: ImageDecoder instance
            fps: Frames per second for output video
        """
        self.image_decoder = image_decoder
        self.fps = fps
        
        print(f"✓ Created VideoDecoder ({fps} fps)")
    
    def decode(
        self,
        frame_token_lists: List[List[int]],
        output_path: str,
        frame_size: tuple = (128, 128),
    ) -> bool:
        """
        Decode frame tokens and stitch into video.
        
        Args:
            frame_token_lists: List of token lists, one per frame
            output_path: Path to save output video
            frame_size: Frame dimensions
        
        Returns:
            True if successful
        """
        if self.image_decoder is None:
            print("⚠ Image decoder not provided")
            return False
        
        if not CV2_AVAILABLE:
            print("⚠ OpenCV not available")
            return False
        
        try:
            # Decode each frame
            frames = []
            for tokens in frame_token_lists:
                frame = self.image_decoder.forward(tokens)
                if frame is not None:
                    # Convert to numpy
                    frame_np = (frame.squeeze(0).permute(1, 2, 0).numpy() * 255).astype('uint8')
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = frame_np[..., ::-1]
                    frames.append(frame_bgr)
            
            if not frames:
                print("✗ No frames to encode")
                return False
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                self.fps,
                frame_size
            )
            
            # Write frames
            for frame in frames:
                writer.write(frame)
            
            writer.release()
            print(f"✓ Saved decoded video to {output_path} ({len(frames)} frames)")
            return True
        
        except Exception as e:
            print(f"✗ Error encoding video: {e}")
            return False


def create_video_decoder(image_decoder=None) -> Optional[VideoDecoder]:
    """Create a video decoder"""
    return VideoDecoder(image_decoder=image_decoder)


if __name__ == "__main__":
    decoder = create_video_decoder()
    if decoder:
        print("✓ Video decoder created")
