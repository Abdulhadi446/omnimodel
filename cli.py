#!/usr/bin/env python3
"""
OmniModel CLI - Command-line interface for the OmniModel system.

Usage:
  python cli.py --input "text here"
  python cli.py --input image.jpg
  python cli.py --input audio.mp3
  python cli.py --input video.mp4
  python cli.py --input document.pdf
  python cli.py --input "draw a cartoon cat" --output-type image
  python cli.py --input "hello" --style creative --think
  python cli.py --input "what is 2+2" --tools

Options:
  --input PATH                Input file or text string (required)
  --style STYLE              Force a specific style (optional)
  --output-type TYPE         Force output type: text, image, audio, video
  --think                    Show reasoning chain
  --tools                    Enable tool calling
  --output OUTPUT            Save output to file
  --help                     Show this help message
"""

import sys
import argparse
import os
from pathlib import Path

# Add parent to path so we can import from omnimodel
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from omnimodel.runtime.router import create_router
    from omnimodel.runtime.loader import create_lazy_loader
    from omnimodel.runtime.tools import create_tool_registry
    from omnimodel.tokenizer.unified import UnifiedTokenizer
    IMPORTS_OK = True
except ImportError as e:
    print(f"⚠ Import error: {e}")
    IMPORTS_OK = False


class OmniModelCLI:
    """Command-line interface for OmniModel"""
    
    def __init__(self):
        """Initialize CLI"""
        self.router = None
        self.loader = None
        self.tools = None
        self.tokenizer = None
        self._init_components()
    
    def _init_components(self):
        """Initialize system components"""
        if not IMPORTS_OK:
            print("⚠ Could not import OmniModel components")
            return
        
        try:
            self.router = create_router()
            self.loader = create_lazy_loader()
            self.tools = create_tool_registry()
            self.tokenizer = UnifiedTokenizer()
            
            print("✓ Initialized OmniModel system")
        except Exception as e:
            print(f"⚠ Could not initialize components: {e}")
    
    def run(
        self,
        input_data: str,
        style: str = None,
        output_type: str = None,
        thinking: bool = False,
        tools: bool = False,
        output_file: str = None,
    ) -> str:
        """
        Run OmniModel inference.
        
        Args:
            input_data: Input text or file path
            style: Optional style override
            output_type: Optional output type override
            thinking: Show reasoning
            tools: Enable tool calling
            output_file: Optional output file path
        
        Returns:
            Result string
        """
        if self.router is None:
            return "ERROR: OmniModel not initialized"
        
        try:
            # Run inference
            result = self.router.infer(
                input_data,
                modality=output_type,
                style=style,
                thinking=thinking,
                tools=tools,
            )
            
            # Save output if requested
            if output_file:
                try:
                    # Determine if result is a file path or text
                    if isinstance(result, str) and Path(result).exists():
                        # Copy the file
                        import shutil
                        shutil.copy(result, output_file)
                        print(f"\n✓ Saved output to {output_file}")
                    else:
                        # Save as text
                        with open(output_file, 'w') as f:
                            f.write(str(result))
                        print(f"\n✓ Saved output to {output_file}")
                except Exception as e:
                    print(f"⚠ Could not save output: {e}")
            
            return str(result)
        
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def print_usage(self):
        """Print usage information"""
        print(__doc__)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="OmniModel - Multimodal AI for 1GB RAM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input text string or file path",
    )
    
    parser.add_argument(
        "--style",
        type=str,
        default=None,
        choices=[
            "human_like", "creative", "professional", "simple", "code",
            "realism", "cartoon", "mcp_handler"
        ],
        help="Output style (optional)",
    )
    
    parser.add_argument(
        "--output-type",
        type=str,
        default=None,
        choices=["text", "image", "audio", "video", "pdf"],
        help="Output type (auto-detected if not specified)",
    )
    
    parser.add_argument(
        "--think",
        action="store_true",
        help="Show reasoning chain",
    )
    
    parser.add_argument(
        "--tools",
        action="store_true",
        help="Enable tool calling",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save output to file",
    )
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = OmniModelCLI()
    
    if cli.router is None:
        print("ERROR: OmniModel not initialized")
        sys.exit(1)
    
    # Run inference
    result = cli.run(
        input_data=args.input,
        style=args.style,
        output_type=args.output_type,
        thinking=args.think,
        tools=args.tools,
        output_file=args.output,
    )
    
    # Print result if not already saved
    if args.output is None and not (isinstance(result, str) and Path(result).exists()):
        print(result)


if __name__ == "__main__":
    main()
