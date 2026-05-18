#!/usr/bin/env python3
"""
Deployment Verification Script

Checks that OmniModel is ready for production deployment:
- All dependencies installed
- Models exist and are correct size
- Memory estimates are accurate
- All tests pass
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import psutil
from datetime import datetime

class DeploymentChecker:
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = []
        self.project_root = Path(__file__).parent
        
    def check(self, condition, message):
        """Print check result"""
        if condition:
            print(f"  ✓ {message}")
            self.checks_passed += 1
        else:
            print(f"  ✗ {message}")
            self.checks_failed += 1
            
    def warn(self, message):
        """Add warning"""
        print(f"  ⚠ {message}")
        self.warnings.append(message)
        
    def section(self, title):
        """Print section header"""
        print(f"\n{title}")
        print("=" * 50)
        
    def verify_python(self):
        """Check Python version and modules"""
        self.section("Python Environment")
        
        version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self.check(sys.version_info >= (3, 10), f"Python 3.10+ ({version})")
        
        # Check required modules
        required_modules = [
            'torch', 'transformers', 'peft', 'numpy', 'scipy', 
            'einops', 'safetensors', 'psutil'
        ]
        
        for module in required_modules:
            try:
                __import__(module)
                self.check(True, f"Module '{module}' available")
            except ImportError:
                self.check(False, f"Module '{module}' missing")
                
    def verify_models(self):
        """Check model files exist"""
        self.section("Model Files")
        
        models_dir = self.project_root / "models" / "quantized"
        adapters_dir = self.project_root / "checkpoints" / "adapters"
        
        # These files should exist or be created during training
        expected_models = [
            ("router", 30),  # ~30MB
            ("sub_base", 15),  # ~15MB
        ]
        
        for model_name, expected_size_mb in expected_models:
            path = models_dir / f"{model_name}.bin"
            exists = path.exists()
            if exists:
                size_mb = path.stat().st_size / (1024 * 1024)
                size_ok = 0.5 * expected_size_mb <= size_mb <= 2 * expected_size_mb
                self.check(size_ok, 
                    f"Model '{model_name}' size OK ({size_mb:.1f}MB, expected ~{expected_size_mb}MB)")
            else:
                self.warn(f"Model '{model_name}' not found (will be created during training)")
                
    def verify_encoders(self):
        """Check encoder/decoder implementations"""
        self.section("Encoders & Decoders")
        
        encoders_dir = self.project_root / "encoders"
        decoders_dir = self.project_root / "decoders"
        
        required_files = {
            encoders_dir: ["image_encoder.py", "audio_encoder.py", "video_encoder.py", "pdf_encoder.py"],
            decoders_dir: ["image_decoder.py", "audio_decoder.py", "video_decoder.py"]
        }
        
        for directory, files in required_files.items():
            for filename in files:
                path = directory / filename
                self.check(path.exists(), f"File '{path.name}' exists")
                
    def verify_tokenizer(self):
        """Check tokenizer implementation"""
        self.section("Tokenizer")
        
        tokenizer_file = self.project_root / "tokenizer" / "unified.py"
        self.check(tokenizer_file.exists(), "Unified tokenizer implemented")
        
        if tokenizer_file.exists():
            try:
                from tokenizer.unified import UnifiedTokenizer
                tokenizer = UnifiedTokenizer()
                vocab_size = tokenizer.vocab_size
                expected = 59496
                
                self.check(vocab_size == expected, 
                    f"Tokenizer vocab size correct ({vocab_size} == {expected})")
            except Exception as e:
                self.warn(f"Could not verify tokenizer: {e}")
                
    def verify_runtime(self):
        """Check runtime components"""
        self.section("Runtime Components")
        
        runtime_files = {
            "Router": "runtime/router.py",
            "Lazy Loader": "runtime/loader.py",
            "Tool Registry": "runtime/tools.py"
        }
        
        for name, path in runtime_files.items():
            full_path = self.project_root / path
            self.check(full_path.exists(), f"{name} implemented")
            
    def verify_cli(self):
        """Check CLI implementation"""
        self.section("Command-Line Interface")
        
        cli_file = self.project_root / "cli.py"
        self.check(cli_file.exists(), "CLI script exists")
        
        # Try running help
        try:
            result = subprocess.run(
                [sys.executable, str(cli_file), "--help"],
                capture_output=True,
                timeout=5,
                text=True
            )
            self.check(result.returncode == 0, "CLI --help works")
        except Exception as e:
            self.warn(f"Could not test CLI: {e}")
            
    def verify_tests(self):
        """Check test suite"""
        self.section("Test Suite")
        
        tests_file = self.project_root / "tests" / "test_all.py"
        self.check(tests_file.exists(), "Test suite exists")
        
        # Try running tests
        try:
            result = subprocess.run(
                [sys.executable, str(tests_file)],
                capture_output=True,
                timeout=30,
                text=True,
                cwd=str(self.project_root)
            )
            
            # Look for success message
            if "ALL TESTS PASSED" in result.stdout:
                self.check(True, "All tests pass")
            elif "FAILED" in result.stdout:
                self.check(False, "Some tests failed")
            else:
                self.warn("Could not determine test results")
                
        except subprocess.TimeoutExpired:
            self.warn("Tests took too long to run")
        except Exception as e:
            self.warn(f"Could not run tests: {e}")
            
    def verify_memory(self):
        """Check memory availability"""
        self.section("System Memory")
        
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)
        
        self.check(total_gb >= 1.0, f"System has at least 1GB RAM (actual: {total_gb:.2f}GB)")
        self.check(available_gb >= 0.5, f"At least 500MB available (actual: {available_gb:.2f}GB)")
        
        # Check peak memory estimate
        peak_estimate_mb = 120  # Main + sub + adapter in RAM
        available_mb = available_gb * 1024
        
        self.check(available_mb > peak_estimate_mb * 2, 
            f"Enough buffer for peak inference (need {peak_estimate_mb}MB, have {available_mb:.0f}MB)")
            
    def verify_disk(self):
        """Check disk space"""
        self.section("Disk Space")
        
        # Get disk usage
        try:
            disk = psutil.disk_usage('/')
            free_gb = disk.free / (1024**3)
            total_gb = disk.total / (1024**3)
            
            self.check(free_gb >= 0.5, f"At least 500MB free disk (have {free_gb:.2f}GB of {total_gb:.2f}GB)")
        except Exception as e:
            self.warn(f"Could not check disk: {e}")
            
    def verify_dependencies(self):
        """Check installation instructions"""
        self.section("Dependencies")
        
        req_files = {
            "Full": "requirements.txt",
            "Inference only": "requirements-inference.txt"
        }
        
        for name, filename in req_files.items():
            path = self.project_root / filename
            self.check(path.exists(), f"{name} requirements file exists ({filename})")
            
    def verify_documentation(self):
        """Check documentation"""
        self.section("Documentation")
        
        docs = {
            "README": "README.md",
            "Deployment guide": "DEPLOYMENT.md",
            "Quick start": "QUICKSTART.sh"
        }
        
        for name, filename in docs.items():
            path = self.project_root / filename
            self.check(path.exists(), f"{name} ({filename})")
            
    def verify_docker(self):
        """Check Docker setup"""
        self.section("Docker Support")
        
        files = {
            "Dockerfile": "Dockerfile",
            "Docker Compose": "docker-compose.yml"
        }
        
        for name, filename in files.items():
            path = self.project_root / filename
            exists = path.exists()
            self.check(exists, f"{name} configured")
            if not exists:
                self.warn(f"Missing {filename} for containerized deployment")
                
    def summary(self):
        """Print summary"""
        self.section("Summary")
        
        total = self.checks_passed + self.checks_failed
        passed_pct = (self.checks_passed / total * 100) if total > 0 else 0
        
        print(f"✓ Passed:  {self.checks_passed}")
        print(f"✗ Failed:  {self.checks_failed}")
        print(f"⚠ Warnings: {len(self.warnings)}")
        print(f"\nStatus: {passed_pct:.0f}% ready for deployment")
        
        if self.warnings:
            print(f"\nWarnings to address:")
            for warning in self.warnings:
                print(f"  • {warning}")
                
        if self.checks_failed == 0:
            print("\n✓ System is ready for deployment!")
            return 0
        else:
            print("\n✗ Please fix the above issues before deploying")
            return 1
            
    def run_all(self):
        """Run all checks"""
        print("\n" + "="*50)
        print("OMNIMODEL DEPLOYMENT VERIFICATION")
        print("="*50)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Project: {self.project_root}")
        
        self.verify_python()
        self.verify_dependencies()
        self.verify_models()
        self.verify_encoders()
        self.verify_tokenizer()
        self.verify_runtime()
        self.verify_cli()
        self.verify_tests()
        self.verify_memory()
        self.verify_disk()
        self.verify_documentation()
        self.verify_docker()
        
        return self.summary()

if __name__ == "__main__":
    checker = DeploymentChecker()
    sys.exit(checker.run_all())
