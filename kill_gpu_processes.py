#!/usr/bin/env python3
"""
GPU'da çalışan Python işlemlerini temizle
"""

import subprocess
import sys
import os

def kill_gpu_processes():
    """GPU'da çalışan Python işlemlerini sonlandır"""
    try:
        # nvidia-smi ile GPU kullanan işlemleri bul
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            pids = [pid.strip() for pid in pids if pid.strip()]
            
            if not pids:
                print("✅ GPU'da çalışan işlem yok")
                return True
            
            print(f"🔍 GPU'da {len(pids)} işlem bulundu")
            
            current_pid = os.getpid()
            for pid in pids:
                try:
                    pid_int = int(pid)
                    if pid_int == current_pid:
                        print(f"⏭️ Kendi işlemimizi atlıyoruz (PID: {pid})")
                        continue
                    
                    # İşlemi sonlandır
                    if sys.platform == 'win32':
                        subprocess.run(['taskkill', '/F', '/PID', pid], 
                                     capture_output=True)
                    else:
                        subprocess.run(['kill', '-9', pid], 
                                     capture_output=True)
                    print(f"❌ İşlem sonlandırıldı (PID: {pid})")
                except Exception as e:
                    print(f"⚠️ İşlem sonlandırılamadı (PID: {pid}): {e}")
            
            return True
        else:
            print("⚠️ nvidia-smi çalıştırılamadı")
            return False
            
    except FileNotFoundError:
        print("⚠️ nvidia-smi bulunamadı")
        return False
    except Exception as e:
        print(f"❌ Hata: {e}")
        return False

if __name__ == "__main__":
    print("🔪 GPU işlemleri temizleniyor...")
    kill_gpu_processes()
    
    # Bellek temizliği
    try:
        import gc
        import torch
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("🧹 CUDA cache temizlendi")
    except:
        pass
