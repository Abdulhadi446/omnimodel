"""
Main Router - Full inference loop that orchestrates the entire system.
Handles: encode input -> route decision -> load sub-model -> generate output -> decode.
"""

import json
from typing import Any, Dict, Optional, Union
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class OmniModelRouter:
    """
    Main inference router that orchestrates the entire OmniModel system.
    Flow:
    1. Encode input (auto-detect modality)
    2. Query main router model for routing decision
    3. Load appropriate sub-model
    4. Generate output from sub-model
    5. Decode output to final format
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: Configuration dict (paths, model names, etc.)
        """
        self.config = config or {}
        
        # Load components (in production)
        self.tokenizer = None
        self.router_model = None
        self.lazy_loader = None
        self.tool_registry = None
        self.encoders = {}
        self.decoders = {}
        self.sub_models = {}
        
        # Current state
        self.current_sub_model = None
        self.thinking_mode = False
        self.tools_enabled = False
        
        print("✓ Created OmniModelRouter")
        
        # Try to initialize components
        self._init_components()
    
    def _init_components(self):
        """Initialize router, sub-models, and other components."""
        if not TORCH_AVAILABLE:
            return
        
        try:
            # Import here to avoid circular imports
            import sys
            import os
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            from tokenizer.unified import UnifiedTokenizer
            from main_model.model import create_router_model
            from sub_models.base import SubModelBase
            from runtime.loader import LazyModelLoader
            from runtime.tools import create_tool_registry
            
            # Initialize tokenizer
            self.tokenizer = UnifiedTokenizer()
            
            # Initialize router model
            self.router_model = create_router_model(pretrained=False)
            
            # Initialize lazy loader
            self.lazy_loader = LazyModelLoader(limit_mb=900)
            
            # Initialize tool registry
            self.tool_registry = create_tool_registry()
            
            # Pre-create one default sub-model for text.general
            base_model = SubModelBase(
                hidden_dim=384,
                num_layers=6,
                num_heads=8,
                vocab_size=59496,
                ffn_dim=1536,
            )
            self.sub_models[('text', 'general')] = base_model
        
        except Exception as e:
            # Silently fail - router will use defaults
            pass
    
    def detect_modality(self, input_data: Union[str, Path]) -> str:
        """
        Auto-detect input modality from file extension or content.
        """
        if isinstance(input_data, str):
            # Check if it looks like a file path (has extension)
            path = Path(input_data)
            suffix = path.suffix.lower()
            
            # Check by extension even if file doesn't exist
            mapping = {
                '.jpg': 'image', '.jpeg': 'image', '.png': 'image',
                '.mp3': 'audio', '.wav': 'audio',
                '.mp4': 'video', '.avi': 'video',
                '.pdf': 'pdf',
            }
            
            if suffix in mapping:
                return mapping[suffix]
            
            # If it has no extension, it's likely text
            return 'text'
        
        return 'text'
    
    def encode_input(self, input_data: Union[str, Path], modality: Optional[str] = None) -> list:
        """
        Encode input to tokens using appropriate encoder.
        """
        if modality is None:
            modality = self.detect_modality(input_data)
        
        print(f"Encoding {modality} input...")
        
        if modality == 'text':
            # Character-level encoding
            if isinstance(input_data, (str, Path)):
                text = str(input_data)
            else:
                text = input_data
            tokens = [ord(c) for c in text]
        
        elif modality == 'image':
            if self.encoders.get('image'):
                tokens = self.encoders['image'].encode_image_file(str(input_data))
            else:
                print("⚠ Image encoder not loaded")
                tokens = []
        
        elif modality == 'audio':
            if self.encoders.get('audio'):
                tokens = self.encoders['audio'].encode(str(input_data))
            else:
                print("⚠ Audio encoder not loaded")
                tokens = []
        
        elif modality == 'video':
            if self.encoders.get('video'):
                frame_tokens = self.encoders['video'].encode(str(input_data))
                tokens = []
                for frame in frame_tokens:
                    tokens.extend(frame)
            else:
                print("⚠ Video encoder not loaded")
                tokens = []
        
        elif modality == 'pdf':
            if self.encoders.get('pdf'):
                tokens = self.encoders['pdf'].encode_to_tokens(str(input_data))
            else:
                print("⚠ PDF encoder not loaded")
                tokens = []
        
        else:
            tokens = []
        
        print(f"  → {len(tokens)} tokens")
        return tokens
    
    def route(self, input_tokens: list) -> Dict[str, str]:
        """
        Use main router model to decide which sub-model to use.
        Returns: {"modality": "text", "style": "creative"}
        """
        print("Routing input...")
        
        if not self.router_model:
            print("⚠ Router model not loaded, using default")
            return {"modality": "text", "style": "general"}
        
        try:
            if TORCH_AVAILABLE:
                # Convert tokens to tensor
                token_tensor = torch.tensor([input_tokens[:128]], dtype=torch.long)  # Limit to 128
                
                # Get router logits
                with torch.no_grad():
                    logits = self.router_model(token_tensor)
                
                # Sample routing decision (simplified)
                # In production, parse the actual route tokens
                return {
                    "modality": "text",
                    "style": "general"
                }
        except Exception as e:
            print(f"⚠ Router error: {e}")
        
        return {"modality": "text", "style": "general"}
    
    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """
        Execute a tool if tools are enabled.
        """
        if not self.tools_enabled:
            return ""
        
        if not self.tool_registry:
            return ""
        
        print(f"Executing tool: {tool_name}")
        result = self.tool_registry.call_tool(tool_name, args)
        print(f"  → {result[:100]}")
        
        return result
    
    def generate(self, input_tokens: list, modality: str, style: str, max_tokens: int = 100) -> list:
        """
        Generate output using appropriate sub-model.
        """
        print(f"Generating {modality}.{style}...")
        
        if not self.sub_models.get((modality, style)):
            print(f"⚠ Sub-model {modality}.{style} not loaded")
            # Return greedy token repetition instead of just echoing
            output_tokens = input_tokens.copy()
            while len(output_tokens) < min(len(input_tokens) + max_tokens, 256):
                # Simple: alternate between input tokens and some generated tokens
                output_tokens.append(input_tokens[len(output_tokens) % len(input_tokens)])
            return output_tokens[:min(len(input_tokens) + max_tokens, 256)]
        
        try:
            sub_model = self.sub_models[(modality, style)]
            
            if TORCH_AVAILABLE and hasattr(sub_model, 'forward'):
                # Generate tokens autoregressively
                input_tensor = torch.tensor([input_tokens[:128]], dtype=torch.long)
                
                output_tokens = input_tokens.copy()
                
                with torch.no_grad():
                    for _ in range(min(max_tokens, 50)):  # Generate up to 50 tokens
                        if len(output_tokens) > 256:
                            break
                        
                        logits = sub_model.forward(input_tensor)
                        if logits is None:
                            break
                        
                        # Sample next token
                        next_logits = logits[0, -1, :]
                        next_token = torch.argmax(next_logits).item()
                        output_tokens.append(next_token)
                        
                        input_tensor = torch.tensor([output_tokens[-128:]], dtype=torch.long)
                
                return output_tokens
        
        except Exception as e:
            print(f"⚠ Generation error: {e}")
        
        return input_tokens[:max_tokens]
    
    def decode_output(self, output_tokens: list, modality: str) -> Any:
        """
        Decode output tokens to final format.
        """
        print(f"Decoding {modality} output...")
        
        if modality == 'text':
            try:
                text = ''.join([chr(t) for t in output_tokens if 0 <= t < 128])
                return text
            except:
                return "".join(str(t) for t in output_tokens[:50])
        
        elif modality == 'image':
            if self.decoders.get('image'):
                output_path = "/tmp/omnmodel_output.jpg"
                self.decoders['image'].decode_to_file(output_tokens, output_path)
                return output_path
        
        elif modality == 'audio':
            if self.decoders.get('audio'):
                output_path = "/tmp/omnimodel_output.wav"
                self.decoders['audio'].decode_to_file(output_tokens, output_path)
                return output_path
        
        else:
            return output_tokens
    
    def infer(
        self,
        input_data: Union[str, Path],
        modality: Optional[str] = None,
        style: Optional[str] = None,
        thinking: bool = False,
        tools: bool = False,
    ) -> str:
        """
        Full inference pipeline.
        """
        print("\n" + "=" * 60)
        print("OMNIMODEL INFERENCE")
        print("=" * 60 + "\n")
        
        self.thinking_mode = thinking
        self.tools_enabled = tools
        
        # Step 1: Encode input
        if modality is None:
            modality = self.detect_modality(input_data)
        
        input_tokens = self.encode_input(input_data, modality)
        
        # Step 2: Route
        route = self.route(input_tokens)
        if style:
            route["style"] = style
        
        print(f"→ Route: {route['modality']}.{route['style']}\n")
        
        # Step 3: Generate
        output_tokens = self.generate(
            input_tokens,
            route['modality'],
            route['style'],
            max_tokens=100
        )
        
        # Step 4: Decode
        output = self.decode_output(output_tokens, route['modality'])
        
        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        print(output)
        
        return str(output)


def create_router() -> OmniModelRouter:
    """Create the main router"""
    return OmniModelRouter()


if __name__ == "__main__":
    router = create_router()
    
    # Test inference
    result = router.infer(
        "Write me a haiku about AI",
        modality="text",
        style="creative"
    )
