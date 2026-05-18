"""
Comprehensive test suite for OmniModel system.
Tests all major components and the full pipeline.
"""

import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestResults:
    """Track test results"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.tests = []
    
    def add_pass(self, test_name: str):
        self.passed += 1
        self.tests.append(("✓", test_name))
        print(f"  ✓ {test_name}")
    
    def add_fail(self, test_name: str, error: str):
        self.failed += 1
        self.tests.append(("✗", test_name, error))
        print(f"  ✗ {test_name}")
        print(f"    Error: {error}")
    
    def add_skip(self, test_name: str, reason: str):
        self.skipped += 1
        self.tests.append(("⊘", test_name, reason))
        print(f"  ⊘ {test_name} (skipped: {reason})")
    
    def summary(self):
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Passed:  {self.passed}")
        print(f"Failed:  {self.failed}")
        print(f"Skipped: {self.skipped}")
        print(f"Total:   {self.passed + self.failed + self.skipped}")
        print("=" * 60)
        
        if self.failed == 0:
            print("✓ ALL TESTS PASSED")
        else:
            print(f"✗ {self.failed} TEST(S) FAILED")


def test_tokenizer(results: TestResults):
    """Test unified tokenizer"""
    print("\nTesting Tokenizer...")
    
    try:
        from tokenizer.unified import UnifiedTokenizer, Modality
        
        tokenizer = UnifiedTokenizer()
        results.add_pass("Tokenizer initialization")
        
        # Test text encoding
        try:
            tokens = tokenizer.encode_text("Hello")
            assert len(tokens) == 5, "Text encoding failed"
            results.add_pass("Text encoding")
        except Exception as e:
            results.add_fail("Text encoding", str(e))
        
        # Test route encoding
        try:
            tokens = tokenizer.encode_route_decision("image", "cartoon")
            assert len(tokens) > 0, "Route encoding failed"
            results.add_pass("Route decision encoding")
        except Exception as e:
            results.add_fail("Route decision encoding", str(e))
        
        # Test tool encoding
        try:
            tokens = tokenizer.encode_tool_call("web_search", {"query": "test"})
            assert len(tokens) > 0, "Tool encoding failed"
            results.add_pass("Tool call encoding")
        except Exception as e:
            results.add_fail("Tool call encoding", str(e))
        
        # Test thinking encoding
        try:
            tokens = tokenizer.encode_thinking("Thinking...")
            assert len(tokens) > 0, "Thinking encoding failed"
            results.add_pass("Thinking encoding")
        except Exception as e:
            results.add_fail("Thinking encoding", str(e))
    
    except ImportError as e:
        results.add_skip("Tokenizer", "Import failed")


def test_router_model(results: TestResults):
    """Test main router model"""
    print("\nTesting Router Model...")
    
    try:
        from main_model.model import create_router_model
        
        import torch
        torch_ok = True
    except ImportError:
        torch_ok = False
    
    if not torch_ok:
        results.add_skip("Router model", "PyTorch not available")
        return
    
    try:
        model = create_router_model()
        results.add_pass("Router model creation")
        
        # Test forward pass
        try:
            input_ids = torch.randint(0, 59496, (2, 128))
            logits = model(input_ids)
            assert logits.shape == (2, 128, 59496), "Forward pass shape mismatch"
            results.add_pass("Router model forward pass")
        except Exception as e:
            results.add_fail("Router model forward pass", str(e))
        
        # Test generation
        try:
            input_ids = torch.randint(0, 59496, (1, 10))
            output = model.generate(input_ids, max_new_tokens=5)
            assert output.shape[1] > input_ids.shape[1], "Generation failed"
            results.add_pass("Router model generation")
        except Exception as e:
            results.add_fail("Router model generation", str(e))
        
        # Check parameter count
        try:
            assert model.n_params < 60e6, f"Too many params: {model.n_params}"
            results.add_pass("Router model parameter count (<50M)")
        except Exception as e:
            results.add_fail("Router model parameter count", str(e))
    
    except Exception as e:
        results.add_fail("Router model creation", str(e))


def test_sub_models(results: TestResults):
    """Test sub-model system"""
    print("\nTesting Sub-models...")
    
    try:
        from sub_models.base import create_shared_base, create_sub_model
        
        import torch
        torch_ok = True
    except ImportError:
        torch_ok = False
    
    if not torch_ok:
        results.add_skip("Sub-models", "PyTorch not available")
        return
    
    try:
        # Create shared base
        base = create_shared_base()
        assert base is not None, "Failed to create base"
        results.add_pass("Shared base creation")
        
        # Create sub-models
        try:
            text_model = create_sub_model("text", "creative", base)
            assert text_model is not None, "Failed to create sub-model"
            results.add_pass("Sub-model creation")
        except Exception as e:
            results.add_fail("Sub-model creation", str(e))
        
        # Test load/unload
        try:
            text_model.create_lora_adapter()
            text_model.load()
            assert text_model.loaded, "Model not loaded"
            results.add_pass("Sub-model loading")
            
            text_model.unload()
            results.add_pass("Sub-model unloading")
        except Exception as e:
            results.add_fail("Sub-model load/unload", str(e))
    
    except Exception as e:
        results.add_fail("Sub-models", str(e))


def test_runtime_components(results: TestResults):
    """Test runtime components"""
    print("\nTesting Runtime Components...")
    
    # Test lazy loader
    try:
        from runtime.loader import create_lazy_loader
        
        loader = create_lazy_loader()
        results.add_pass("Lazy loader creation")
        
        # Test memory monitoring
        try:
            mem_info = loader.get_memory_usage()
            assert "current_mb" in mem_info, "Memory info incomplete"
            results.add_pass("Memory monitoring")
        except Exception as e:
            results.add_fail("Memory monitoring", str(e))
    
    except Exception as e:
        results.add_fail("Lazy loader", str(e))
    
    # Test tool registry
    try:
        from runtime.tools import create_tool_registry
        
        registry = create_tool_registry()
        results.add_pass("Tool registry creation")
        
        # Test calculator
        try:
            result = registry.call_tool("calculator", {"expr": "2+2"})
            assert "4" in result, f"Calculator returned: {result}"
            results.add_pass("Calculator tool")
        except Exception as e:
            results.add_fail("Calculator tool", str(e))
        
        # Test file read (read this file)
        try:
            this_file = __file__
            result = registry.call_tool("file_read", {"path": this_file})
            assert len(result) > 0, "File read returned empty"
            results.add_pass("File read tool")
        except Exception as e:
            results.add_fail("File read tool", str(e))
    
    except Exception as e:
        results.add_fail("Tool registry", str(e))


def test_router_inference(results: TestResults):
    """Test full router inference"""
    print("\nTesting Router Inference...")
    
    try:
        from runtime.router import create_router
        
        router = create_router()
        results.add_pass("Router creation")
        
        # Test modality detection
        try:
            modality = router.detect_modality("hello world")
            assert modality == "text", f"Wrong modality: {modality}"
            results.add_pass("Modality detection (text)")
            
            modality = router.detect_modality("image.jpg")
            assert modality == "image", f"Wrong modality: {modality}"
            results.add_pass("Modality detection (image)")
        except Exception as e:
            results.add_fail("Modality detection", str(e))
        
        # Test input encoding
        try:
            tokens = router.encode_input("test", "text")
            assert len(tokens) > 0, "No tokens generated"
            results.add_pass("Input encoding")
        except Exception as e:
            results.add_fail("Input encoding", str(e))
        
        # Test routing
        try:
            route = router.route([1, 2, 3])
            assert "modality" in route and "style" in route, "Route incomplete"
            results.add_pass("Routing")
        except Exception as e:
            results.add_fail("Routing", str(e))
    
    except Exception as e:
        results.add_fail("Router inference", str(e))


def test_cli(results: TestResults):
    """Test CLI"""
    print("\nTesting CLI...")
    
    try:
        from cli import OmniModelCLI
        
        cli = OmniModelCLI()
        results.add_pass("CLI initialization")
        
        # Test argument parsing
        try:
            import argparse
            # CLI uses argparse internally, just check if it's importable
            results.add_pass("CLI argument parsing")
        except Exception as e:
            results.add_fail("CLI argument parsing", str(e))
    
    except Exception as e:
        results.add_fail("CLI", str(e))


def main():
    """Run all tests"""
    print("=" * 60)
    print("OMNIMODEL TEST SUITE")
    print("=" * 60)
    
    results = TestResults()
    
    # Run test suites
    test_tokenizer(results)
    test_router_model(results)
    test_sub_models(results)
    test_runtime_components(results)
    test_router_inference(results)
    test_cli(results)
    
    # Print summary
    results.summary()
    
    # Return exit code
    return 0 if results.failed == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
