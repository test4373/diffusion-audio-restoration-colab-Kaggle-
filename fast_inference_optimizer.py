# ---------------------------------------------------------------
# Fast Inference Optimizer for PyTorch Models
# Hızlı inference için PyTorch optimizasyonları
# ---------------------------------------------------------------

import torch
import torch.nn as nn
from typing import Optional, Callable
import warnings
import os


class FastInferenceOptimizer:
    """
    PyTorch modelleri için hız optimizasyonu sağlar.
    
    Özellikler:
    - torch.compile() desteği (PyTorch 2.0+)
    - Mixed precision (FP16/BF16)
    - CUDA optimizasyonları
    - Model quantization (INT8)
    - Inference mode optimizasyonu
    """
    
    def __init__(
        self,
        use_compile: bool = True,
        use_mixed_precision: bool = True,
        precision: str = "fp16",  # "fp16", "bf16", "fp32"
        use_cudnn_benchmark: bool = True,
        use_tf32: bool = True,
        compile_mode: str = "reduce-overhead",  # "default", "reduce-overhead", "max-autotune"
        quantize: bool = False,
        device: Optional[str] = None
    ):
        """
        Args:
            use_compile: torch.compile() kullan (PyTorch 2.0+)
            use_mixed_precision: Mixed precision kullan
            precision: Precision tipi ("fp16", "bf16", "fp32")
            use_cudnn_benchmark: cuDNN benchmark modu
            use_tf32: TF32 kullan (Ampere+ GPU'lar için)
            compile_mode: Compile modu
            quantize: INT8 quantization kullan
            device: Device ("cuda", "cpu", None=auto)
        """
        self.use_compile = use_compile
        self.use_mixed_precision = use_mixed_precision
        self.precision = precision
        self.use_cudnn_benchmark = use_cudnn_benchmark
        self.use_tf32 = use_tf32
        self.compile_mode = compile_mode
        self.quantize = quantize
        
        # Device ayarla
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # PyTorch versiyonu kontrol et
        self.torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        self.supports_compile = self.torch_version >= (2, 0)
        
        if self.use_compile and not self.supports_compile:
            warnings.warn(
                f"torch.compile() requires PyTorch 2.0+, but found {torch.__version__}. "
                "Disabling compile optimization."
            )
            self.use_compile = False
            
        # BF16 desteği kontrol et
        if self.precision == "bf16" and self.device == "cuda":
            if not torch.cuda.is_bf16_supported():
                warnings.warn(
                    "BF16 not supported on this GPU. Falling back to FP16."
                )
                self.precision = "fp16"
                
        self._setup_cuda_optimizations()
        
    def _setup_cuda_optimizations(self):
        """CUDA optimizasyonlarını ayarla"""
        if self.device == "cuda":
            # cuDNN benchmark
            if self.use_cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
                print("✓ cuDNN benchmark enabled")
                
            # TF32 (Ampere+ GPU'lar için)
            if self.use_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("✓ TF32 enabled")
                
            # CUDA cache ayarları
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """
        Modeli optimize et
        
        Args:
            model: Optimize edilecek PyTorch modeli
            
        Returns:
            Optimize edilmiş model
        """
        print(f"\n{'='*60}")
        print("🚀 Fast Inference Optimizer")
        print(f"{'='*60}")
        
        # Model'i eval moduna al
        model.eval()
        
        # Device'a taşı
        model = model.to(self.device)
        print(f"✓ Model moved to {self.device}")
        
        # Mixed precision
        if self.use_mixed_precision and self.device == "cuda":
            if self.precision == "fp16":
                model = model.half()
                print("✓ FP16 (half precision) enabled")
            elif self.precision == "bf16":
                model = model.bfloat16()
                print("✓ BF16 (bfloat16 precision) enabled")
        
        # Quantization
        if self.quantize:
            model = self._quantize_model(model)
            
        # torch.compile()
        if self.use_compile:
            print(f"✓ Compiling model with mode='{self.compile_mode}'...")
            try:
                model = torch.compile(model, mode=self.compile_mode)
                print("✓ Model compiled successfully")
            except Exception as e:
                warnings.warn(f"Failed to compile model: {e}")
                
        print(f"{'='*60}\n")
        return model
    
    def _quantize_model(self, model: nn.Module) -> nn.Module:
        """Model'i INT8'e quantize et"""
        try:
            # Dynamic quantization (en hızlı ve kolay)
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
            print("✓ INT8 quantization applied")
            return quantized_model
        except Exception as e:
            warnings.warn(f"Quantization failed: {e}")
            return model
    
    @staticmethod
    def create_inference_context(
        use_amp: bool = True,
        dtype: torch.dtype = torch.float16
    ):
        """
        Inference için context manager oluştur
        
        Usage:
            with optimizer.create_inference_context():
                output = model(input)
        """
        class InferenceContext:
            def __init__(self, use_amp, dtype):
                self.use_amp = use_amp
                self.dtype = dtype
                
            def __enter__(self):
                self.inference_mode = torch.inference_mode()
                self.inference_mode.__enter__()
                
                if self.use_amp and torch.cuda.is_available():
                    self.autocast = torch.cuda.amp.autocast(dtype=self.dtype)
                    self.autocast.__enter__()
                else:
                    self.autocast = None
                    
                return self
                
            def __exit__(self, *args):
                if self.autocast is not None:
                    self.autocast.__exit__(*args)
                self.inference_mode.__exit__(*args)
                
        return InferenceContext(use_amp, dtype)
    
    def get_optimal_batch_size(
        self,
        model: nn.Module,
        input_shape: tuple,
        max_memory_gb: float = None
    ) -> int:
        """
        GPU memory'ye göre optimal batch size hesapla
        
        Args:
            model: PyTorch modeli
            input_shape: Input tensor shape (C, H, W)
            max_memory_gb: Maksimum kullanılacak memory (GB)
            
        Returns:
            Optimal batch size
        """
        if self.device != "cuda":
            return 1
            
        if max_memory_gb is None:
            # Toplam GPU memory'nin %80'ini kullan
            total_memory = torch.cuda.get_device_properties(0).total_memory
            max_memory_gb = (total_memory * 0.8) / (1024**3)
            
        # Test batch size'ları
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        optimal_batch_size = 8
        
        for bs in batch_sizes:
            try:
                # Test input oluştur
                test_input = torch.randn(bs, *input_shape).to(self.device)
                if self.precision == "fp16":
                    test_input = test_input.half()
                elif self.precision == "bf16":
                    test_input = test_input.bfloat16()
                    
                # Forward pass
                with torch.inference_mode():
                    _ = model(test_input)
                    
                # Memory kullanımını kontrol et
                memory_used = torch.cuda.max_memory_allocated() / (1024**3)
                
                if memory_used < max_memory_gb:
                    optimal_batch_size = bs
                else:
                    break
                    
                # Memory'yi temizle
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    break
                raise e
                
        print(f"✓ Optimal batch size: {optimal_batch_size}")
        return optimal_batch_size


class FastDiffusionSampler:
    """
    Diffusion modelleri için hızlandırılmış sampling
    """
    
    def __init__(
        self,
        optimizer: FastInferenceOptimizer,
        use_ddim: bool = True,
        ddim_eta: float = 0.0
    ):
        """
        Args:
            optimizer: FastInferenceOptimizer instance
            use_ddim: DDIM sampling kullan (daha hızlı)
            ddim_eta: DDIM eta parametresi (0=deterministic)
        """
        self.optimizer = optimizer
        self.use_ddim = use_ddim
        self.ddim_eta = ddim_eta
        
    def sample_fast(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t_steps: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Hızlı sampling
        
        Args:
            model: Diffusion model
            x_t: Noisy input
            t_steps: Time steps
            **kwargs: Ek parametreler
            
        Returns:
            Denoised output
        """
        with self.optimizer.create_inference_context():
            if self.use_ddim:
                return self._ddim_sample(model, x_t, t_steps, **kwargs)
            else:
                return self._ddpm_sample(model, x_t, t_steps, **kwargs)
                
    def _ddim_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t_steps: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """DDIM sampling (daha hızlı)"""
        # DDIM implementation
        # Bu kısım mevcut ddpm_sample fonksiyonuna entegre edilebilir
        pass
        
    def _ddpm_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t_steps: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Standard DDPM sampling"""
        # Standard DDPM implementation
        pass


def benchmark_model(
    model: nn.Module,
    input_shape: tuple,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device: str = "cuda"
) -> dict:
    """
    Model performansını benchmark et
    
    Args:
        model: PyTorch modeli
        input_shape: Input shape (B, C, H, W)
        num_iterations: Benchmark iterasyon sayısı
        warmup_iterations: Warmup iterasyon sayısı
        device: Device
        
    Returns:
        Benchmark sonuçları
    """
    import time
    
    model = model.to(device).eval()
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Warmup
    print(f"Warming up ({warmup_iterations} iterations)...")
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            _ = model(dummy_input)
            
    if device == "cuda":
        torch.cuda.synchronize()
        
    # Benchmark
    print(f"Benchmarking ({num_iterations} iterations)...")
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
    
    results = {
        "total_time": total_time,
        "avg_time_per_iteration": avg_time,
        "throughput": throughput,
        "device": device
    }
    
    print(f"\n{'='*60}")
    print("Benchmark Results:")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.4f}s")
    print(f"Avg time per iteration: {avg_time*1000:.2f}ms")
    print(f"Throughput: {throughput:.2f} iterations/s")
    print(f"{'='*60}\n")
    
    return results


# Kullanım örneği
if __name__ == "__main__":
    print("Fast Inference Optimizer - Usage Example\n")
    
    # Optimizer oluştur
    optimizer = FastInferenceOptimizer(
        use_compile=True,
        use_mixed_precision=True,
        precision="fp16",
        compile_mode="reduce-overhead"
    )
    
    # Örnek model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 3, 3, padding=1)
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.conv3(x)
            return x
    
    model = DummyModel()
    
    # Model'i optimize et
    optimized_model = optimizer.optimize_model(model)
    
    # Optimal batch size bul
    optimal_bs = optimizer.get_optimal_batch_size(
        optimized_model,
        input_shape=(3, 256, 256)
    )
    
    # Benchmark
    if torch.cuda.is_available():
        benchmark_model(
            optimized_model,
            input_shape=(optimal_bs, 3, 256, 256),
            num_iterations=100
        )
