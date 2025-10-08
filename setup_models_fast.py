#!/usr/bin/env python3
"""
HIZLI Model İndirme - Kaggle için optimize edilmiş
Alternatif yöntemler:
1. Kaggle Dataset kullanımı (ÖNERİLEN - saniyeler içinde)
2. wget ile paralel indirme (daha hızlı)
3. HuggingFace CLI (daha güvenilir)
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path

# Kaggle ortamını tespit et
IN_KAGGLE = os.path.exists('/kaggle/working')

if IN_KAGGLE:
    SCRIPT_DIR = '/kaggle/working/diffusion-audio-restoration-colab-Kaggle-'
    CKPT_DIR = '/kaggle/working/ckpt'
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    CKPT_DIR = os.path.join(SCRIPT_DIR, 'ckpt')

# Model bilgileri
MODELS = {
    'A2SB_onesplit_0.0_1.0_release.ckpt': 
        'https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge/resolve/main/ckpt/A2SB_onesplit_0.0_1.0_release.ckpt',
    'A2SB_twosplit_0.5_1.0_release.ckpt':
        'https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge/resolve/main/ckpt/A2SB_twosplit_0.5_1.0_release.ckpt'
}

def print_header(text):
    """Başlık yazdır"""
    print("\n" + "="*70)
    print(text)
    print("="*70)

def check_existing_models():
    """Mevcut modelleri kontrol et"""
    os.makedirs(CKPT_DIR, exist_ok=True)
    
    existing = []
    missing = []
    
    for model_name in MODELS.keys():
        model_path = os.path.join(CKPT_DIR, model_name)
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            existing.append((model_name, size_mb))
        else:
            missing.append(model_name)
    
    return existing, missing

def method_1_kaggle_dataset():
    """Yöntem 1: Kaggle Dataset kullan (EN HIZLI - ÖNERİLEN)"""
    print_header("🚀 YÖNTEM 1: Kaggle Dataset (ÖNERİLEN)")
    
    print("\n📦 Bu yöntem modelleri Kaggle Dataset'inden kopyalar (saniyeler içinde!)")
    print("\n⚠️ ÖNCELİKLE YAPMANIZ GEREKENLER:")
    print("   1. Kaggle notebook'unuzda sağ tarafta 'Add Data' butonuna tıklayın")
    print("   2. 'nvidia/audio-to-audio-schrodinger-bridge' dataset'ini arayın")
    print("   3. Dataset'i notebook'unuza ekleyin")
    print("   4. Aşağıdaki komutu çalıştırın:\n")
    
    print("# Kaggle notebook'unuzda bu komutu çalıştırın:")
    print("-" * 70)
    print("!mkdir -p /kaggle/working/ckpt")
    print("!cp /kaggle/input/audio-to-audio-schrodinger-bridge/ckpt/*.ckpt /kaggle/working/ckpt/")
    print("-" * 70)
    
    print("\n💡 Bu yöntem 5-10 saniye içinde tamamlanır!")
    print("💡 İnternet hızınızdan bağımsızdır!")
    
    return False

def method_2_wget_parallel():
    """Yöntem 2: wget ile paralel indirme (HIZLI)"""
    print_header("🚀 YÖNTEM 2: wget ile Paralel İndirme")
    
    print("\n📥 Bu yöntem wget kullanarak daha hızlı indirir")
    print("⏱️ Tahmini süre: 3-5 dakika (internet hızına bağlı)\n")
    
    try:
        # wget kurulu mu kontrol et
        result = subprocess.run(['which', 'wget'], capture_output=True)
        if result.returncode != 0:
            print("❌ wget bulunamadı. Yükleniyor...")
            subprocess.run(['apt-get', 'install', '-y', 'wget'], check=True)
        
        print("✅ wget hazır\n")
        
        # Her model için wget komutu
        for model_name, url in MODELS.items():
            model_path = os.path.join(CKPT_DIR, model_name)
            
            if os.path.exists(model_path):
                print(f"⏭️ {model_name} zaten mevcut, atlanıyor...")
                continue
            
            print(f"📥 İndiriliyor: {model_name}")
            
            # wget ile indir (progress bar ile)
            cmd = [
                'wget',
                '--progress=bar:force:noscroll',
                '--tries=3',
                '--timeout=30',
                '-O', model_path,
                url
            ]
            
            result = subprocess.run(cmd)
            
            if result.returncode == 0:
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                print(f"✅ İndirildi: {model_name} ({size_mb:.1f} MB)\n")
            else:
                print(f"❌ İndirme başarısız: {model_name}\n")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        return False

def method_3_huggingface_cli():
    """Yöntem 3: HuggingFace CLI (GÜVENİLİR)"""
    print_header("🚀 YÖNTEM 3: HuggingFace CLI")
    
    print("\n📥 Bu yöntem HuggingFace CLI kullanır (daha güvenilir)")
    print("⏱️ Tahmini süre: 5-8 dakika\n")
    
    try:
        # huggingface-cli kurulu mu kontrol et
        result = subprocess.run(['which', 'huggingface-cli'], capture_output=True)
        if result.returncode != 0:
            print("📦 huggingface-hub yükleniyor...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface-hub[cli]'], check=True)
        
        print("✅ HuggingFace CLI hazır\n")
        
        # Modelleri indir
        for model_name in MODELS.keys():
            model_path = os.path.join(CKPT_DIR, model_name)
            
            if os.path.exists(model_path):
                print(f"⏭️ {model_name} zaten mevcut, atlanıyor...")
                continue
            
            print(f"📥 İndiriliyor: {model_name}")
            
            cmd = [
                'huggingface-cli',
                'download',
                'nvidia/audio_to_audio_schrodinger_bridge',
                f'ckpt/{model_name}',
                '--local-dir', CKPT_DIR,
                '--local-dir-use-symlinks', 'False'
            ]
            
            result = subprocess.run(cmd)
            
            if result.returncode == 0:
                # Dosyayı doğru konuma taşı
                downloaded_path = os.path.join(CKPT_DIR, 'ckpt', model_name)
                if os.path.exists(downloaded_path):
                    os.rename(downloaded_path, model_path)
                    # Boş ckpt klasörünü sil
                    try:
                        os.rmdir(os.path.join(CKPT_DIR, 'ckpt'))
                    except:
                        pass
                
                if os.path.exists(model_path):
                    size_mb = os.path.getsize(model_path) / (1024 * 1024)
                    print(f"✅ İndirildi: {model_name} ({size_mb:.1f} MB)\n")
                else:
                    print(f"❌ Dosya bulunamadı: {model_name}\n")
                    return False
            else:
                print(f"❌ İndirme başarısız: {model_name}\n")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        return False

def method_4_aria2():
    """Yöntem 4: aria2c ile çok hızlı indirme (EN HIZLI İNDİRME)"""
    print_header("🚀 YÖNTEM 4: aria2c ile Süper Hızlı İndirme")
    
    print("\n⚡ Bu yöntem aria2c kullanarak çok hızlı indirir")
    print("⏱️ Tahmini süre: 2-4 dakika (internet hızına bağlı)\n")
    
    try:
        # aria2c kurulu mu kontrol et
        result = subprocess.run(['which', 'aria2c'], capture_output=True)
        if result.returncode != 0:
            print("📦 aria2 yükleniyor...")
            subprocess.run(['apt-get', 'install', '-y', 'aria2'], check=True)
        
        print("✅ aria2c hazır\n")
        
        # Her model için aria2c komutu
        for model_name, url in MODELS.items():
            model_path = os.path.join(CKPT_DIR, model_name)
            
            if os.path.exists(model_path):
                print(f"⏭️ {model_name} zaten mevcut, atlanıyor...")
                continue
            
            print(f"📥 İndiriliyor: {model_name}")
            
            # aria2c ile indir (16 bağlantı, çok hızlı)
            cmd = [
                'aria2c',
                '--max-connection-per-server=16',
                '--split=16',
                '--min-split-size=1M',
                '--file-allocation=none',
                '--continue=true',
                '--dir', CKPT_DIR,
                '--out', model_name,
                url
            ]
            
            result = subprocess.run(cmd)
            
            if result.returncode == 0:
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                print(f"✅ İndirildi: {model_name} ({size_mb:.1f} MB)\n")
            else:
                print(f"❌ İndirme başarısız: {model_name}\n")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        return False

def update_config():
    """Konfigürasyonu güncelle"""
    print_header("⚙️ Konfigürasyon Güncelleniyor")
    
    checkpoint_paths = [
        os.path.join(CKPT_DIR, model_name)
        for model_name in MODELS.keys()
    ]
    
    config_path = os.path.join(SCRIPT_DIR, 'configs', 'ensemble_2split_sampling.yaml')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config['model']['pretrained_checkpoints'] = checkpoint_paths
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Konfigürasyon güncellendi: {config_path}")
        for i, path in enumerate(checkpoint_paths, 1):
            print(f"   {i}. {path}")
        
        return True
    except Exception as e:
        print(f"❌ Hata: {e}")
        return False

def main():
    print_header("🎵 A2SB HIZLI Model Kurulumu")
    
    if IN_KAGGLE:
        print("✅ Kaggle ortamı tespit edildi")
    else:
        print("💻 Yerel ortam tespit edildi")
    
    print(f"📁 Checkpoint dizini: {CKPT_DIR}")
    
    # Mevcut modelleri kontrol et
    existing, missing = check_existing_models()
    
    if existing:
        print("\n✅ Mevcut modeller:")
        for name, size in existing:
            print(f"   • {name}: {size:.1f} MB")
    
    if not missing:
        print("\n🎉 Tüm modeller zaten mevcut!")
        update_config()
        return 0
    
    print(f"\n❌ Eksik modeller ({len(missing)}):")
    for name in missing:
        print(f"   • {name}")
    
    # Yöntem seçimi
    print("\n" + "="*70)
    print("📋 İNDİRME YÖNTEMLERİ")
    print("="*70)
    print("\n1. 🚀 Kaggle Dataset (ÖNERİLEN - saniyeler içinde)")
    print("2. ⚡ aria2c (EN HIZLI - 2-4 dakika)")
    print("3. 📥 wget (HIZLI - 3-5 dakika)")
    print("4. 🔒 HuggingFace CLI (GÜVENİLİR - 5-8 dakika)")
    print("5. ❌ İptal")
    
    choice = input("\nYöntem seçin (1-5): ").strip()
    
    success = False
    
    if choice == '1':
        method_1_kaggle_dataset()
        print("\n💡 Yukarıdaki adımları tamamladıktan sonra bu scripti tekrar çalıştırın.")
        return 0
    elif choice == '2':
        success = method_4_aria2()
    elif choice == '3':
        success = method_2_wget_parallel()
    elif choice == '4':
        success = method_3_huggingface_cli()
    elif choice == '5':
        print("\n❌ İptal edildi.")
        return 1
    else:
        print("\n❌ Geçersiz seçim!")
        return 1
    
    if success:
        # Konfigürasyonu güncelle
        if update_config():
            print_header("✅ KURULUM TAMAMLANDI!")
            print("\n🚀 Şimdi Gradio uygulamasını başlatabilirsiniz:")
            print("   !python gradio_app_kaggle.py --ngrok")
            return 0
        else:
            print("\n⚠️ Modeller indirildi ama konfigürasyon güncellenemedi.")
            return 1
    else:
        print("\n❌ İndirme başarısız oldu. Başka bir yöntem deneyin.")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n❌ İptal edildi.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
