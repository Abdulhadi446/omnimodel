"""
Unified Tokenizer - Handles all modalities with special tokens.
One unified token space for text, images, audio, video, and PDFs.
"""

import json
from typing import List, Dict, Any
from enum import Enum


class Modality(Enum):
    """Supported modalities"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    PDF = "pdf"


class UnifiedTokenizer:
    """
    Unified tokenizer that converts all modalities to a single token space.
    Uses special tokens for modality boundaries and control sequences.
    """
    
    def __init__(self, base_vocab_size: int = 50257):
        """
        Initialize tokenizer with special tokens.
        
        Args:
            base_vocab_size: Size of base vocabulary (e.g., GPT-2 has 50257)
        """
        self.base_vocab_size = base_vocab_size
        self.special_tokens = {}
        self.token_to_id = {}
        self.id_to_token = {}
        
        # Build special token vocabulary
        self._build_special_tokens()
    
    def _build_special_tokens(self):
        """Define and register all special tokens"""
        
        # Modality markers
        modality_tokens = [
            "<text>", "</text>",
            "<image>", "</image>",
            "<audio>", "</audio>",
            "<video>", "</video>",
            "<pdf>", "</pdf>",
        ]
        
        # Reasoning and control tokens
        control_tokens = [
            "<think>", "</think>",
            "<tool>", "</tool>",
            "<args>", "</args>",
            "<result>", "</result>",
            "<route>", "</route>",
            "<style>", "</style>",
            "<frame_sep>",
        ]
        
        # Image tokens (VQVAE codes: 8192 possible)
        image_tokens = [f"<img_tok_{i}>" for i in range(8192)]
        
        # Audio tokens (EnCodec codes: 1024 possible)
        audio_tokens = [f"<aud_tok_{i}>" for i in range(1024)]
        
        # Combine all special tokens
        all_special = modality_tokens + control_tokens + image_tokens + audio_tokens
        
        # Assign IDs starting after base vocab
        next_id = self.base_vocab_size
        for token in all_special:
            token_id = next_id
            self.special_tokens[token] = token_id
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
            next_id += 1
        
        self.vocab_size = next_id
        
        print(f"✓ Built unified tokenizer")
        print(f"  Base vocab size: {self.base_vocab_size}")
        print(f"  Special tokens: {len(self.special_tokens)}")
        print(f"  Total vocab size: {self.vocab_size}")
    
    def encode_text(self, text: str) -> List[int]:
        """
        Encode text input.
        For now, use simple character-level encoding.
        In production, use BPE tokenizer.
        """
        tokens = [ord(c) for c in text]
        return tokens
    
    def encode_with_modality(self, content: Any, modality: Modality) -> List[int]:
        """
        Encode content with modality wrapper.
        
        Args:
            content: The content to encode
            modality: The modality type
            
        Returns:
            List of token IDs including modality markers
        """
        tokens = []
        
        # Add opening modality marker
        open_token = f"<{modality.value}>"
        tokens.append(self.special_tokens[open_token])
        
        # Encode content based on modality
        if modality == Modality.TEXT:
            tokens.extend(self.encode_text(content))
        elif modality == Modality.IMAGE:
            # Expect list of image token indices (0-8191)
            if isinstance(content, list):
                tokens.extend([
                    self.special_tokens[f"<img_tok_{idx}>"]
                    for idx in content
                ])
        elif modality == Modality.AUDIO:
            # Expect list of audio token indices (0-1023)
            if isinstance(content, list):
                tokens.extend([
                    self.special_tokens[f"<aud_tok_{idx}>"]
                    for idx in content
                ])
        elif modality == Modality.VIDEO:
            # Expect list of frame token sequences separated by <frame_sep>
            if isinstance(content, list):
                for i, frame_tokens in enumerate(content):
                    if i > 0:
                        tokens.append(self.special_tokens["<frame_sep>"])
                    tokens.extend([
                        self.special_tokens[f"<img_tok_{idx}>"]
                        for idx in frame_tokens
                    ])
        elif modality == Modality.PDF:
            # Expect mixed content (text and image tokens)
            if isinstance(content, dict):
                for item in content.get("content", []):
                    if item["type"] == "text":
                        tokens.extend(self.encode_text(item["value"]))
                    elif item["type"] == "image":
                        tokens.extend([
                            self.special_tokens[f"<img_tok_{idx}>"]
                            for idx in item["value"]
                        ])
        
        # Add closing modality marker
        close_token = f"</{modality.value}>"
        tokens.append(self.special_tokens[close_token])
        
        return tokens
    
    def encode_route_decision(self, modality: str, style: str) -> List[int]:
        """
        Encode a router decision: <route>modality.style</route>
        """
        tokens = []
        tokens.append(self.special_tokens["<route>"])
        
        # Route string as text tokens
        route_str = f"{modality}.{style}"
        tokens.extend(self.encode_text(route_str))
        
        tokens.append(self.special_tokens["</route>"])
        return tokens
    
    def encode_tool_call(self, tool_name: str, args: Dict[str, Any]) -> List[int]:
        """
        Encode a tool call: <tool>name</tool><args>{"key": "value"}</args>
        """
        tokens = []
        tokens.append(self.special_tokens["<tool>"])
        tokens.extend(self.encode_text(tool_name))
        tokens.append(self.special_tokens["</tool>"])
        
        tokens.append(self.special_tokens["<args>"])
        args_json = json.dumps(args)
        tokens.extend(self.encode_text(args_json))
        tokens.append(self.special_tokens["</args>"])
        
        return tokens
    
    def encode_thinking(self, thought: str) -> List[int]:
        """
        Encode reasoning: <think>...</think>
        """
        tokens = []
        tokens.append(self.special_tokens["<think>"])
        tokens.extend(self.encode_text(thought))
        tokens.append(self.special_tokens["</think>"])
        return tokens
    
    def decode_tokens(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        Only handles text tokens and special tokens for now.
        """
        result = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                result.append(self.id_to_token[token_id])
            elif token_id < self.base_vocab_size:
                # Character-level decode
                result.append(chr(token_id))
            else:
                result.append(f"<unk_{token_id}>")
        return "".join(result)
    
    def get_special_token_id(self, token: str) -> int:
        """Get ID for a special token"""
        return self.special_tokens.get(token, None)
    
    def is_special_token(self, token_id: int) -> bool:
        """Check if token is special"""
        return token_id >= self.base_vocab_size


# Example usage
if __name__ == "__main__":
    tokenizer = UnifiedTokenizer()
    
    # Example: encode text input
    text_tokens = tokenizer.encode_with_modality(
        "Hello, world!",
        Modality.TEXT
    )
    print(f"\nText tokens: {text_tokens[:10]}...")  # Show first 10
    
    # Example: encode route decision
    route_tokens = tokenizer.encode_route_decision("image", "cartoon")
    print(f"Route tokens: {route_tokens}")
    
    # Example: encode tool call
    tool_tokens = tokenizer.encode_tool_call(
        "web_search",
        {"query": "what is AI"}
    )
    print(f"Tool tokens: {tool_tokens}")
    
    # Example: encode thinking
    think_tokens = tokenizer.encode_thinking(
        "The user wants a cartoon image"
    )
    print(f"Thinking tokens: {think_tokens}")
