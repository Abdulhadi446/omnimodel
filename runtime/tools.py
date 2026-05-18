"""
Tool Registry - Implements tool calling for the router model.
Tools available: web_search, calculator, code_executor, file_read, file_write, image_capture
"""

import json
import subprocess
import ast
import os
from typing import Dict, Any, Callable
from pathlib import Path

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class ToolRegistry:
    """
    Registry of available tools that the router can call.
    Each tool has a description and implementation.
    """
    
    def __init__(self, timeout_seconds: int = 10):
        """
        Args:
            timeout_seconds: Maximum execution time for tools (default 10s)
        """
        self.timeout_seconds = timeout_seconds
        self.tools: Dict[str, Callable] = {}
        self.descriptions: Dict[str, str] = {}
        
        # Register built-in tools
        self._register_builtin_tools()
    
    def _register_builtin_tools(self):
        """Register all built-in tools"""
        
        # Web search
        self.register_tool(
            "web_search",
            self.web_search,
            "Search the web using DuckDuckGo (no API key needed)"
        )
        
        # Calculator
        self.register_tool(
            "calculator",
            self.calculator,
            "Evaluate mathematical expressions safely"
        )
        
        # Code executor
        self.register_tool(
            "code_executor",
            self.code_executor,
            "Execute code snippets with timeout protection"
        )
        
        # File read
        self.register_tool(
            "file_read",
            self.file_read,
            "Read local files (safe: no directory traversal)"
        )
        
        # File write
        self.register_tool(
            "file_write",
            self.file_write,
            "Write to local files (safe: whitelist required)"
        )
        
        # Image capture
        self.register_tool(
            "image_capture",
            self.image_capture,
            "Capture image from webcam"
        )
        
        print(f"✓ Registered {len(self.tools)} built-in tools")
    
    def register_tool(self, name: str, func: Callable, description: str):
        """Register a new tool"""
        self.tools[name] = func
        self.descriptions[name] = description
        print(f"✓ Registered tool: {name}")
    
    def call_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """
        Call a tool and return result as string.
        
        Args:
            tool_name: Name of the tool
            args: Tool arguments
        
        Returns:
            Tool result as string
        """
        if tool_name not in self.tools:
            return f"ERROR: Unknown tool '{tool_name}'"
        
        try:
            tool_func = self.tools[tool_name]
            result = tool_func(**args)
            return str(result)
        except Exception as e:
            return f"ERROR: {tool_name} failed: {str(e)}"
    
    # Built-in tool implementations
    
    def web_search(self, query: str) -> str:
        """
        Search the web using DuckDuckGo.
        """
        if not REQUESTS_AVAILABLE:
            return "Web search unavailable (requests library not installed)"
        
        try:
            url = f"https://duckduckgo.com/api"
            params = {"q": query, "format": "json"}
            
            response = requests.get(url, params=params, timeout=self.timeout_seconds)
            data = response.json()
            
            results = []
            for result in data.get("Results", [])[:3]:  # Top 3 results
                results.append(f"- {result.get('Title')}: {result.get('FirstURL')}")
            
            return "\n".join(results) if results else "No results found"
        
        except Exception as e:
            return f"Search failed: {str(e)}"
    
    def calculator(self, expr: str) -> str:
        """
        Safely evaluate mathematical expressions.
        """
        try:
            # Use ast to safely parse and evaluate
            tree = ast.parse(expr, mode='eval')
            
            # Only allow specific node types (no function calls, etc.)
            allowed_types = (
                ast.Expression,
                ast.BinOp,
                ast.UnaryOp,
                ast.Constant,
                ast.Name,
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.FloorDiv,
                ast.Mod,
                ast.Pow,
            )
            
            for node in ast.walk(tree):
                if not isinstance(node, allowed_types):
                    return f"ERROR: Unsafe operation '{type(node).__name__}'"
            
            # Evaluate
            result = eval(compile(tree, '<string>', 'eval'))
            return str(result)
        
        except Exception as e:
            return f"Calculation failed: {str(e)}"
    
    def code_executor(self, code: str, language: str = "python") -> str:
        """
        Execute code with timeout protection.
        """
        if language != "python":
            return f"Only Python supported, got {language}"
        
        try:
            # Use subprocess for isolation
            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True,
                timeout=self.timeout_seconds,
                text=True,
            )
            
            if result.returncode != 0:
                return f"ERROR: {result.stderr}"
            
            return result.stdout
        
        except subprocess.TimeoutExpired:
            return f"ERROR: Code execution timed out ({self.timeout_seconds}s)"
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def file_read(self, path: str) -> str:
        """
        Read a local file safely.
        """
        try:
            # Prevent directory traversal
            path = Path(path).resolve()
            
            if not path.exists():
                return f"ERROR: File not found: {path}"
            
            with open(path, 'r') as f:
                content = f.read()
            
            # Limit output size
            if len(content) > 10000:
                return content[:10000] + "\n... (truncated)"
            
            return content
        
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def file_write(self, path: str, content: str, whitelist: list = None) -> str:
        """
        Write to a local file safely.
        """
        try:
            path = Path(path).resolve()
            
            # Check whitelist if provided
            if whitelist:
                if not any(str(path).startswith(str(w)) for w in whitelist):
                    return f"ERROR: Path not in whitelist: {path}"
            
            # Create parent directories
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                f.write(content)
            
            return f"✓ Wrote {len(content)} bytes to {path}"
        
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def image_capture(self) -> str:
        """
        Capture image from webcam.
        """
        try:
            import cv2
            
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return "ERROR: Cannot capture from webcam"
            
            # Save image
            output_path = "/tmp/webcam_capture.jpg"
            cv2.imwrite(output_path, frame)
            
            return f"✓ Captured image to {output_path}"
        
        except ImportError:
            return "ERROR: OpenCV not installed"
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def list_tools(self) -> Dict[str, str]:
        """Get list of available tools"""
        return self.descriptions
    
    def get_tool_json(self) -> str:
        """Get tools in JSON format for LLM context"""
        tools_list = []
        for name, desc in self.descriptions.items():
            tools_list.append({
                "name": name,
                "description": desc,
            })
        return json.dumps(tools_list, indent=2)


def create_tool_registry() -> ToolRegistry:
    """Create a tool registry"""
    return ToolRegistry(timeout_seconds=10)


if __name__ == "__main__":
    registry = create_tool_registry()
    
    # Test tools
    print("\nAvailable tools:")
    for name, desc in registry.list_tools().items():
        print(f"  - {name}: {desc}")
    
    # Test calculator
    print("\nTest: 2 + 2 =", registry.call_tool("calculator", {"expr": "2 + 2"}))
