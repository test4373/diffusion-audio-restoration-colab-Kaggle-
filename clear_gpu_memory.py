#!/usr/bin/env python3
"""
GPU belleğini temizleme scripti
Inference öncesi çalıştırılarak bellek sorunlarını önler
"""

import gc
import torch

def clear_gpu_memory():
    """GPU belleğini temizle"""
    try:
        # Python garbage collector'ı çalıştır
        gc.collect()
        
        # CUDA cache'i temizle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Bellek istatistiklerini göster
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                
                print(f"GPU {i}:")
                print(f"  Allocated: {allocated:.2f} GB")
                print(f"  Reserved: {reserved:.2f} GB")
                print(f"  Total: {total:.2f} GB")
                print(f"  Free: {total - allocated:.2f} GB")
            
            return True
        else:
            print("CUDA kullanılamıyor")
            return False
            
    except Exception as e:
        print(f"GPU bellek temizleme hatası: {e}")
        return False

if __name__ == "__main__":
    print("🧹 GPU belleği temizleniyor...")
    success = clear_gpu_memory()
    if success:
        print("✅ GPU belleği temizlendi")
    else:
        print("❌ GPU bellek temizleme başarısız")
