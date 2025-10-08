#!/usr/bin/env python3
# ---------------------------------------------------------------
# Fast Inference Test Script
# PyTorch optimizasyonlarını test eder
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import time
import argparse
from fast_inference_optimizer import FastInferenceOptimizer, benchmark_model


class SimpleTestModel(nn.Module):
    """Test için basit bir model"""
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv4 = nn.Conv2d(channels, 3, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x


def test_basic_optimization():
    """Temel optimizasyon testleri"""
    print("\n" + "="*60)
    print("TEST 1: Basic Optimization")
    print("="*60)
    
    # Model oluştur
    model = SimpleTestModel(channels=64)
    
    # Optimizer oluştur
    optimizer = FastInferenceOptimizer(
        use_compile=True,
        use_mixed_precision=True,
        precision="fp16",
        compile_mode="reduce-overhead"
    )
    
    # Model'i optimize et
    optimized_model = optimizer.optimize_model(model)
    
    print("✓ Basic optimization test passed!")
    return optimized_model


def test_inference_speed(model, input_shape=(1, 3, 256, 256), num_iterations=100):
    """Inference hızını test et"""
    print("\n" + "="*60)
    print("TEST 2: Inference Speed")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    
    # Test input
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Warmup
    print("Warming up...")
    with torch.inference_mode():
        for _ in range(10):
            _ = model(dummy_input)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running {num_iterations} iterations...")
    start_time = time.time()
    
    with torch.inference_mode():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    # Sonuçlar
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    throughput = num_iterations / total_time
    
    print(f"\nResults:")
    print(f"  Total time: {total_time:.4f}s")
    print(f"  Avg time per iteration: {avg_time*1000:.2f}ms")
    print(f"  Throughput: {throughput:.2f} iterations/s")
    print(f"  Device: {device}")
    
    print("✓ Inference speed test passed!")
    return avg_time


def test_precision_modes():
    """Farklı precision modlarını test et"""
    print("\n" + "="*60)
    print("TEST 3: Precision Modes")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, skipping precision tests")
        return
    
    model = SimpleTestModel(channels=32)
    input_shape = (1, 3, 128, 128)
    
    precisions = ["fp32", "fp16"]
    if torch.cuda.is_bf16_supported():
        precisions.append("bf16")
    
    results = {}
    
    for precision in precisions:
        print(f"\nTesting {precision}...")
        
        optimizer = FastInferenceOptimizer(
            use_compile=False,  # Compile olmadan test et
            use_mixed_precision=(precision != "fp32"),
            precision=precision
        )
        
        optimized_model = optimizer.optimize_model(model)
        avg_time = test_inference_speed(optimized_model, input_shape, num_iterations=50)
        results[precision] = avg_time
    
    # Karşılaştırma
    print("\n" + "-"*60)
    print("Precision Comparison:")
    print("-"*60)
    baseline = results["fp32"]
    for precision, avg_time in results.items():
        speedup = baseline / avg_time
        print(f"{precision:6s}: {avg_time*1000:6.2f}ms (speedup: {speedup:.2f}x)")
    
    print("✓ Precision modes test passed!")


def test_compile_modes():
    """Farklı compile modlarını test et"""
    print("\n" + "="*60)
    print("TEST 4: Compile Modes")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, skipping compile tests")
        return
    
    # PyTorch versiyonu kontrol et
    torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
    if torch_version < (2, 0):
        print(f"⚠️ torch.compile() requires PyTorch 2.0+, found {torch.__version__}")
        return
    
    model = SimpleTestModel(channels=32)
    input_shape = (1, 3, 128, 128)
    
    compile_modes = ["default", "reduce-overhead", "max-autotune"]
    results = {}
    
    # Baseline (no compile)
    print("\nTesting without compile...")
    optimizer = FastInferenceOptimizer(use_compile=False, use_mixed_precision=True, precision="fp16")
    optimized_model = optimizer.optimize_model(model)
    baseline_time = test_inference_speed(optimized_model, input_shape, num_iterations=50)
    results["no_compile"] = baseline_time
    
    # Test compile modes
    for mode in compile_modes:
        print(f"\nTesting compile mode: {mode}...")
        
        optimizer = FastInferenceOptimizer(
            use_compile=True,
            use_mixed_precision=True,
            precision="fp16",
            compile_mode=mode
        )
        
        optimized_model = optimizer.optimize_model(model)
        avg_time = test_inference_speed(optimized_model, input_shape, num_iterations=50)
        results[mode] = avg_time
    
    # Karşılaştırma
    print("\n" + "-"*60)
    print("Compile Mode Comparison:")
    print("-"*60)
    for mode, avg_time in results.items():
        speedup = baseline_time / avg_time
        print(f"{mode:20s}: {avg_time*1000:6.2f}ms (speedup: {speedup:.2f}x)")
    
    print("✓ Compile modes test passed!")


def test_memory_usage():
    """Memory kullanımını test et"""
    print("\n" + "="*60)
    print("TEST 5: Memory Usage")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, skipping memory tests")
        return
    
    model = SimpleTestModel(channels=64)
    input_shape = (4, 3, 256, 256)  # Larger batch
    
    precisions = ["fp32", "fp16"]
    
    for precision in precisions:
        print(f"\nTesting {precision}...")
        
        # Reset memory stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        optimizer = FastInferenceOptimizer(
            use_compile=False,
            use_mixed_precision=(precision != "fp32"),
            precision=precision
        )
        
        optimized_model = optimizer.optimize_model(model)
        dummy_input = torch.randn(*input_shape).to("cuda")
        
        # Forward pass
        with torch.inference_mode():
            _ = optimized_model(dummy_input)
        
        # Memory stats
        memory_allocated = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        memory_reserved = torch.cuda.max_memory_reserved() / (1024**2)  # MB
        
        print(f"  Memory allocated: {memory_allocated:.2f} MB")
        print(f"  Memory reserved: {memory_reserved:.2f} MB")
    
    print("✓ Memory usage test passed!")


def test_optimal_batch_size():
    """Optimal batch size testi"""
    print("\n" + "="*60)
    print("TEST 6: Optimal Batch Size")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, skipping batch size test")
        return
    
    model = SimpleTestModel(channels=64)
    
    optimizer = FastInferenceOptimizer(
        use_compile=False,
        use_mixed_precision=True,
        precision="fp16"
    )
    
    optimized_model = optimizer.optimize_model(model)
    
    # Optimal batch size bul
    optimal_bs = optimizer.get_optimal_batch_size(
        optimized_model,
        input_shape=(3, 256, 256),
        max_memory_gb=None  # Auto-detect
    )
    
    print(f"\n✓ Optimal batch size: {optimal_bs}")
    print("✓ Optimal batch size test passed!")


def run_all_tests():
    """Tüm testleri çalıştır"""
    print("\n" + "="*60)
    print("🚀 FAST INFERENCE OPTIMIZER - TEST SUITE")
    print("="*60)
    
    # System info
    print("\nSystem Information:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    try:
        # Test 1: Basic optimization
        optimized_model = test_basic_optimization()
        
        # Test 2: Inference speed
        test_inference_speed(optimized_model)
        
        # Test 3: Precision modes
        test_precision_modes()
        
        # Test 4: Compile modes
        test_compile_modes()
        
        # Test 5: Memory usage
        test_memory_usage()
        
        # Test 6: Optimal batch size
        test_optimal_batch_size()
        
        # Summary
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60 + "\n")
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ TEST FAILED: {e}")
        print("="*60 + "\n")
        raise


def main():
    parser = argparse.ArgumentParser(description='Test Fast Inference Optimizer')
    parser.add_argument('--test', type=str, default='all',
                       choices=['all', 'basic', 'speed', 'precision', 'compile', 'memory', 'batch'],
                       help='Which test to run')
    args = parser.parse_args()
    
    if args.test == 'all':
        run_all_tests()
    elif args.test == 'basic':
        test_basic_optimization()
    elif args.test == 'speed':
        model = SimpleTestModel()
        test_inference_speed(model)
    elif args.test == 'precision':
        test_precision_modes()
    elif args.test == 'compile':
        test_compile_modes()
    elif args.test == 'memory':
        test_memory_usage()
    elif args.test == 'batch':
        test_optimal_batch_size()


if __name__ == "__main__":
    main()
