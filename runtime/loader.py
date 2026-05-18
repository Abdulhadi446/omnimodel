"""
Lazy Model Loader - Manages loading/unloading of models to stay under 1GB RAM.
Keeps main model always in RAM, maintains cache of 1 sub-model.
"""

from typing import Optional, Dict, Any
import psutil
import os

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class LazyModelLoader:
    """
    Efficiently loads/unloads models to stay within RAM constraints.
    Strategy:
    - Main router model: always in RAM (~25MB)
    - Shared sub-model base: always in RAM (~10MB)
    - Active sub-model adapter: in RAM (~5MB)
    - Encoders/decoders: loaded on-demand (~80MB each)
    Total active: ~120MB (well under 1GB)
    """
    
    def __init__(self, ram_limit_mb: int = 900):
        """
        Args:
            ram_limit_mb: Maximum RAM to use (default 900MB, leaving 100MB buffer)
        """
        self.ram_limit_mb = ram_limit_mb
        self.current_loaded_model = None
        self.model_cache = {}
        self.loaded_adapters = {}
        
        print(f"✓ Created LazyModelLoader (limit: {ram_limit_mb}MB)")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get current memory usage.
        """
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            memory_mb = memory_info.rss / (1024 * 1024)
            
            return {
                "current_mb": round(memory_mb, 1),
                "limit_mb": self.ram_limit_mb,
                "percent": round(memory_mb / self.ram_limit_mb * 100, 1),
                "available": memory_mb < self.ram_limit_mb,
            }
        except Exception as e:
            print(f"⚠ Could not get memory usage: {e}")
            return {
                "current_mb": 0,
                "limit_mb": self.ram_limit_mb,
                "percent": 0,
                "available": True,
            }
    
    def load_model(self, model_key: str, model: Any, force: bool = False) -> bool:
        """
        Load a model, unloading others if necessary.
        
        Args:
            model_key: Unique identifier for the model
            model: Model instance
            force: Force load even if over limit
        
        Returns:
            True if successful
        """
        # Check memory
        mem_info = self.get_memory_usage()
        
        if mem_info["current_mb"] > self.ram_limit_mb and not force:
            print(f"⚠ Memory limit exceeded: {mem_info['current_mb']}MB / {self.ram_limit_mb}MB")
            print("  Unloading previous model...")
            self.unload_model()
        
        # Load model
        if model is None:
            print(f"✗ Cannot load None model: {model_key}")
            return False
        
        try:
            if TORCH_AVAILABLE:
                model.cpu()  # Ensure on CPU
            
            self.current_loaded_model = model_key
            self.model_cache[model_key] = model
            
            mem_info = self.get_memory_usage()
            print(f"✓ Loaded {model_key} ({mem_info['current_mb']}MB / {self.ram_limit_mb}MB)")
            
            return True
        
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    
    def unload_model(self, model_key: Optional[str] = None) -> bool:
        """
        Unload a model from RAM.
        """
        key = model_key or self.current_loaded_model
        
        if key is None:
            return False
        
        try:
            if key in self.model_cache:
                model = self.model_cache[key]
                
                if TORCH_AVAILABLE:
                    try:
                        model.cpu()
                        if hasattr(torch, 'cuda'):
                            torch.cuda.empty_cache()
                    except:
                        pass
                
                del self.model_cache[key]
                
                if self.current_loaded_model == key:
                    self.current_loaded_model = None
                
                print(f"✓ Unloaded {key}")
                return True
        
        except Exception as e:
            print(f"✗ Error unloading model: {e}")
            return False
        
        return False
    
    def switch_model(self, new_model_key: str, new_model: Any) -> bool:
        """
        Switch to a different model (unload current, load new).
        """
        if self.current_loaded_model is not None:
            self.unload_model()
        
        return self.load_model(new_model_key, new_model)
    
    def preload_adapter(self, adapter_key: str, adapter_path: Optional[str] = None) -> bool:
        """
        Preload a LoRA adapter weights.
        """
        try:
            if TORCH_AVAILABLE and adapter_path:
                weights = torch.load(adapter_path)
                self.loaded_adapters[adapter_key] = weights
                print(f"✓ Preloaded adapter: {adapter_key}")
                return True
        except Exception as e:
            print(f"✗ Error preloading adapter: {e}")
        
        return False
    
    def estimate_model_size(self, model: Any) -> float:
        """
        Estimate model size in MB.
        """
        try:
            if TORCH_AVAILABLE:
                param_count = sum(p.numel() for p in model.parameters())
                # Rough estimate: 4 bytes per parameter (FP32)
                size_mb = param_count * 4 / (1024 * 1024)
                return size_mb
        except:
            pass
        
        return 0.0
    
    def clear_cache(self) -> None:
        """Clear all cached models"""
        for key in list(self.model_cache.keys()):
            self.unload_model(key)
        
        self.model_cache.clear()
        self.loaded_adapters.clear()
        
        print("✓ Cleared model cache")


def create_lazy_loader(ram_limit_mb: int = 900) -> LazyModelLoader:
    """Create a lazy model loader"""
    return LazyModelLoader(ram_limit_mb=ram_limit_mb)


if __name__ == "__main__":
    loader = create_lazy_loader()
    
    # Test memory monitoring
    mem_info = loader.get_memory_usage()
    print(f"\nMemory status:")
    print(f"  Current: {mem_info['current_mb']}MB")
    print(f"  Limit: {mem_info['limit_mb']}MB")
    print(f"  Usage: {mem_info['percent']}%")
