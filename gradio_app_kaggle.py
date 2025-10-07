import gradio as gr
import os
import sys
import yaml
import librosa
import numpy as np
from datetime import datetime
import argparse

# ============================================================================
# KAGGLE ORTAM AYARLARI
# ============================================================================

# Kaggle ortamını tespit et
IN_KAGGLE = os.path.exists('/kaggle/working')

if IN_KAGGLE:
    # Kaggle'da çalışma dizini
    SCRIPT_DIR = '/kaggle/working/diffusion-audio-restoration-colab-Kaggle-'
    OUTPUT_DIR = '/kaggle/working/gradio_outputs'
    print("🌐 Kaggle ortamı tespit edildi")
else:
    # Yerel ortam
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    OUTPUT_DIR = "gradio_outputs"
    print("💻 Yerel ortam")

# Çalışma dizinini ayarla
os.chdir(SCRIPT_DIR)

# Python path'e ekle
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

print(f"📁 Çalışma Dizini: {SCRIPT_DIR}")
print(f"📁 Çıktı Dizini: {OUTPUT_DIR}")
print(f"🐍 Python Path: {sys.path[:3]}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# MODÜLLERI İÇE AKTAR
# ============================================================================

try:
    from A2SB_lightning_module_api import TimePartitionedPretrainedSTFTBridgeModel
    from datasets.datamodule import STFTAudioDataModule
    print("✅ Modüller başarıyla yüklendi")
except ImportError as e:
    print(f"❌ Modül yükleme hatası: {e}")
    print("Lütfen doğru dizinde olduğunuzdan emin olun")

# ============================================================================
# YARDIMCI FONKSİYONLAR
# ============================================================================

def compute_rolloff_freq(audio_file, roll_percent=0.99):
    """Otomatik cutoff frekansı tespiti"""
    try:
        y, sr = librosa.load(audio_file, sr=None)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=roll_percent)[0]
        return int(np.mean(rolloff))
    except Exception as e:
        print(f"Rolloff hesaplama hatası: {e}")
        return 2000

def kill_gpu_processes():
    """GPU'da çalışan diğer Python işlemlerini sonlandır"""
    try:
        import subprocess
        
        # nvidia-smi ile GPU kullanan işlemleri bul
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            pids = [pid.strip() for pid in pids if pid.strip()]
            
            current_pid = os.getpid()
            for pid in pids:
                try:
                    pid_int = int(pid)
                    if pid_int != current_pid:
                        # İşlemi sonlandır (Linux/Kaggle)
                        subprocess.run(['kill', '-9', pid], 
                                     capture_output=True, check=False)
                        print(f"❌ GPU işlemi sonlandırıldı (PID: {pid})")
                except:
                    pass
    except:
        pass

def clear_gpu_memory():
    """GPU belleğini temizle"""
    try:
        import gc
        import torch
        
        # Önce diğer işlemleri temizle
        kill_gpu_processes()
        
        # Python garbage collector
        gc.collect()
        
        if torch.cuda.is_available():
            # Tüm GPU'ları temizle
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            
            torch.cuda.synchronize()
            print("🧹 GPU belleği temizlendi")
            return True
        return False
    except Exception as e:
        print(f"⚠️ GPU bellek temizleme hatası: {e}")
        return False

def run_inference_direct(config_files, n_steps, output_path):
    """Inference'ı CLI ile çalıştır - subprocess kullanarak"""
    try:
        import subprocess
        
        # GPU belleğini temizle
        clear_gpu_memory()
        
        # PYTHONPATH'i environment variable olarak ayarla
        env = os.environ.copy()
        env['PYTHONPATH'] = SCRIPT_DIR
        
        # CUDA bellek optimizasyonları
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
        env['CUDA_LAUNCH_BLOCKING'] = '0'
        
        # Kaggle için özel ayarlar
        if IN_KAGGLE:
            env['KAGGLE_KERNEL_RUN_TYPE'] = 'Interactive'
        
        # Python'ı -u (unbuffered) flag'i ile çalıştır
        cmd = [
            sys.executable,
            '-u',  # Unbuffered output
            os.path.join(SCRIPT_DIR, 'ensembled_inference_api.py'),
            'predict',
            '-c', config_files[0],
            '-c', config_files[1],
            f'--model.predict_n_steps={n_steps}',
            f'--model.output_audio_filename={output_path}',
            '--trainer.strategy=auto',
            '--trainer.devices=1',
            '--trainer.accelerator=auto',
            '--trainer.precision=16-mixed',  # Mixed precision için bellek tasarrufu
            '--data.batch_size=1'  # Batch size'ı 1'e düşür
        ]
        
        print(f"🔧 Komut çalıştırılıyor...")
        print(f"📁 PYTHONPATH: {env['PYTHONPATH']}")
        print(f"📁 Çalışma dizini: {SCRIPT_DIR}")
        print(f"📄 Config 1: {config_files[0]}")
        print(f"📄 Config 2: {config_files[1]}")
        print(f"📤 Çıktı: {output_path}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=SCRIPT_DIR,
            env=env,
            timeout=1800  # 30 dakika timeout (Kaggle için)
        )
        
        if result.returncode == 0:
            print(f"✅ Inference tamamlandı!")
            return True, None
        else:
            # Daha fazla hata detayı göster
            error_msg = f"Hata kodu: {result.returncode}\n\n"
            error_msg += f"STDOUT (son 2000 karakter):\n{result.stdout[-2000:]}\n\n"
            error_msg += f"STDERR (son 2000 karakter):\n{result.stderr[-2000:]}"
            print(f"❌ {error_msg}")
            return False, error_msg
        
    except subprocess.TimeoutExpired:
        error_msg = "⏱️ İşlem zaman aşımına uğradı (30 dakika). Daha kısa ses dosyaları kullanın."
        print(f"❌ {error_msg}")
        return False, error_msg
    except Exception as e:
        import traceback
        error_msg = f"Hata: {str(e)}\n\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        return False, error_msg

def restore_audio(audio_file, mode, n_steps, cutoff_freq_auto, cutoff_freq_manual, 
                  inpaint_length):
    """Ana ses restorasyon fonksiyonu"""
    try:
        print("🚀 Başlatılıyor...")
        
        if audio_file is None:
            return None, "❌ Lütfen bir ses dosyası yükleyin!"
        
        # Dosya adı oluştur
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_filename = os.path.basename(audio_file)
        base_name = input_filename.rsplit('.', 1)[0] if '.' in input_filename else input_filename
        output_filename = f"{timestamp}_{mode}_{base_name}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        print("📊 Ses dosyası analiz ediliyor...")
        
        # Ses bilgilerini al
        y, sr = librosa.load(audio_file, sr=None)
        duration = len(y) / sr
        
        info_text = f"## 📊 Ses Bilgileri\n\n"
        info_text += f"- **Sampling Rate:** {sr} Hz\n"
        info_text += f"- **Duration:** {duration:.2f} saniye\n"
        info_text += f"- **Samples:** {len(y):,}\n\n"
        
        # Kaggle için süre uyarısı
        if IN_KAGGLE and duration > 60:
            info_text += f"⚠️ **Uyarı:** Ses dosyası {duration:.1f} saniye. Kaggle'da uzun işlemler zaman aşımına uğrayabilir.\n\n"
        
        # Config dosyalarını hazırla
        base_config = os.path.join(SCRIPT_DIR, 'configs', 'ensemble_2split_sampling.yaml')
        
        if mode == "bandwidth":
            print("🎯 Bandwidth extension hazırlanıyor...")
            
            # Cutoff frekansını belirle
            if cutoff_freq_auto:
                cutoff_freq = compute_rolloff_freq(audio_file)
                info_text += f"🎯 **Otomatik Cutoff:** {cutoff_freq} Hz\n"
            else:
                cutoff_freq = int(cutoff_freq_manual)
                info_text += f"🎯 **Manuel Cutoff:** {cutoff_freq} Hz\n"
            
            info_text += f"⚙️ **Sampling Steps:** {n_steps}\n\n"
            
            # Config hazırla
            config_path = os.path.join(SCRIPT_DIR, 'configs', 'inference_files_upsampling.yaml')
            if not os.path.exists(config_path):
                return None, f"❌ Config dosyası bulunamadı: {config_path}"
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            config['data']['predict_filelist'] = [{
                'filepath': audio_file,
                'output_subdir': '.'
            }]
            
            config['data']['transforms_aug'][0]['init_args']['upsample_mask_kwargs'] = {
                'min_cutoff_freq': cutoff_freq,
                'max_cutoff_freq': cutoff_freq
            }
            
            # Bellek optimizasyonu: segment_length'i azalt
            if 'segment_length' in config['data']:
                config['data']['segment_length'] = 65280  # ~1.5s @ 44.1kHz
                info_text += f"⚙️ **Segment Length:** 65280 samples (~1.5s) - Bellek optimizasyonu\n"
            
            temp_config = os.path.join(SCRIPT_DIR, 'configs', f'temp_gradio_{timestamp}.yaml')
            with open(temp_config, 'w') as f:
                yaml.dump(config, f)
        
        elif mode == "inpainting":
            print("🎨 Audio inpainting hazırlanıyor...")
            
            info_text += f"🎯 **Inpainting Length:** {inpaint_length}s\n"
            info_text += f"⚙️ **Sampling Steps:** {n_steps}\n\n"
            
            # Config hazırla
            config_path = os.path.join(SCRIPT_DIR, 'configs', 'inference_files_inpainting.yaml')
            if not os.path.exists(config_path):
                return None, f"❌ Config dosyası bulunamadı: {config_path}"
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            config['data']['predict_filelist'] = [{
                'filepath': audio_file,
                'output_subdir': '.'
            }]
            
            config['data']['transforms_aug'][0]['init_args']['inpainting_mask_kwargs'] = {
                'min_inpainting_frac': inpaint_length,
                'max_inpainting_frac': inpaint_length,
                'is_random': False
            }
            
            temp_config = os.path.join(SCRIPT_DIR, 'configs', f'temp_gradio_{timestamp}.yaml')
            with open(temp_config, 'w') as f:
                yaml.dump(config, f)
        
        print("🔄 Model çalıştırılıyor... (Bu birkaç dakika sürebilir)")
        
        # Inference çalıştır
        success, error = run_inference_direct([base_config, temp_config], n_steps, output_path)
        
        # Geçici config'i sil
        if os.path.exists(temp_config):
            os.remove(temp_config)
        
        if not success:
            return None, f"## ❌ Hata\n\n```\n{error}\n```"
        
        print("✨ Tamamlanıyor...")
        
        # Çıktı dosyasını kontrol et
        if os.path.exists(output_path):
            info_text += f"---\n\n## ✅ İşlem Tamamlandı!\n\n"
            info_text += f"📁 **Çıktı Dosyası:** `{output_filename}`\n"
            
            if IN_KAGGLE:
                info_text += f"📁 **Kaggle Yolu:** `/kaggle/working/gradio_outputs/{output_filename}`\n\n"
            else:
                info_text += f"\n"
            
            # Çıktı ses bilgileri
            y_out, sr_out = librosa.load(output_path, sr=None)
            info_text += f"## 📊 Restore Edilmiş Ses\n\n"
            info_text += f"- **Sampling Rate:** {sr_out} Hz\n"
            info_text += f"- **Duration:** {len(y_out)/sr_out:.2f} saniye\n"
            info_text += f"- **Samples:** {len(y_out):,}\n\n"
            
            # Spektral özellikler
            cent_orig = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
            cent_rest = np.mean(librosa.feature.spectral_centroid(y=y_out, sr=sr_out)[0])
            rolloff_orig = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99)[0])
            rolloff_rest = np.mean(librosa.feature.spectral_rolloff(y=y_out, sr=sr_out, roll_percent=0.99)[0])
            
            info_text += f"## 📈 Spektral Analiz\n\n"
            info_text += f"| Özellik | Orijinal | Restored | Değişim |\n"
            info_text += f"|---------|----------|----------|---------|\\n"
            info_text += f"| Spectral Centroid | {cent_orig:.0f} Hz | {cent_rest:.0f} Hz | {((cent_rest-cent_orig)/cent_orig*100):+.1f}% |\n"
            info_text += f"| Spectral Rolloff | {rolloff_orig:.0f} Hz | {rolloff_rest:.0f} Hz | {((rolloff_rest-rolloff_orig)/rolloff_orig*100):+.1f}% |\n"
            
            print("✅ Tamamlandı!")
            return output_path, info_text
        else:
            return None, info_text + "\n---\n\n## ❌ Hata\n\nÇıktı dosyası oluşturulamadı."
    
    except Exception as e:
        import traceback
        error_msg = f"## ❌ Hata Oluştu\n\n```\n{str(e)}\n\n{traceback.format_exc()}\n```"
        return None, error_msg

# ============================================================================
# GRADIO ARAYÜZÜ
# ============================================================================

# Kaggle için özel tema ve ayarlar
custom_css = """
.gradio-container {
    max-width: 1200px !important;
}
"""

with gr.Blocks(title="A2SB Audio Restoration - Kaggle", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("""
    # 🎵 A2SB: Audio-to-Audio Schrödinger Bridge
    ### Yüksek Kaliteli Ses Restorasyonu - NVIDIA
    
    """ + ("🌐 **Kaggle Edition** - Ücretsiz GPU ile ses restorasyonu!" if IN_KAGGLE else "�� **Local Edition**") + """
    
    Ses dosyalarınızı AI ile restore edin!
    """)
    
    if IN_KAGGLE:
        gr.Markdown("""
        ### 📌 Kaggle Kullanım Notları:
        - ⏱️ **İşlem Süresi:** 10 saniyelik ses için ~2-3 dakika (P100 GPU)
        - 💾 **Çıktılar:** `/kaggle/working/gradio_outputs/` dizinine kaydedilir
        - 📥 **İndirme:** Arayüzden veya Output sekmesinden indirebilirsiniz
        - ⚠️ **Limit:** Çok uzun ses dosyaları (>60s) zaman aşımına uğrayabilir
        """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Giriş")
            
            audio_input = gr.Audio(
                label="Ses Dosyası Yükle",
                type="filepath",
                sources=["upload", "microphone"]
            )
            
            mode = gr.Radio(
                choices=["bandwidth", "inpainting"],
                value="bandwidth",
                label="Restorasyon Modu",
                info="Bandwidth: Yüksek frekansları restore et | Inpainting: Eksik kısımları doldur"
            )
            
            with gr.Accordion("⚙️ Gelişmiş Ayarlar", open=False):
                n_steps = gr.Slider(
                    25, 100, 50, step=5, 
                    label="Sampling Steps",
                    info="Daha yüksek = daha iyi kalite ama daha yavaş"
                )
                cutoff_freq_auto = gr.Checkbox(
                    True, 
                    label="Otomatik Cutoff",
                    info="Cutoff frekansını otomatik tespit et"
                )
                cutoff_freq_manual = gr.Slider(
                    1000, 10000, 2000, step=100, 
                    label="Manuel Cutoff (Hz)", 
                    visible=False,
                    info="Bandwidth extension için cutoff frekansı"
                )
                inpaint_length = gr.Slider(
                    0.1, 1.0, 0.3, step=0.1, 
                    label="Inpainting Uzunluğu (s)",
                    info="Doldurulacak eksik kısmın uzunluğu"
                )
            
            cutoff_freq_auto.change(
                fn=lambda x: gr.update(visible=not x),
                inputs=[cutoff_freq_auto],
                outputs=[cutoff_freq_manual]
            )
            
            restore_btn = gr.Button("🚀 Restore Et", variant="primary", size="lg")
            
            if IN_KAGGLE:
                gr.Markdown("""
                ### 💡 İpuçları:
                - İlk çalıştırmada model yükleme ~1-2 dakika sürer
                - Kısa ses dosyalarıyla test edin (10-30s)
                - P100 GPU seçin (Settings > Accelerator)
                """)
        
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Çıktı")
            audio_output = gr.Audio(label="Restore Edilmiş Ses", type="filepath")
            info_output = gr.Markdown("Ses dosyası yükleyin ve 'Restore Et' butonuna tıklayın.")
    
    # Örnekler - Kaggle için devre dışı (JSON schema hatası nedeniyle)
    # if IN_KAGGLE:
    #     gr.Markdown("### 📚 Örnek Kullanım")
    #     gr.Examples(
    #         examples=[
    #             ["bandwidth", 50, True, 2000, 0.3],
    #             ["bandwidth", 75, False, 4000, 0.3],
    #             ["inpainting", 50, True, 2000, 0.5],
    #         ],
    #         inputs=[mode, n_steps, cutoff_freq_auto, cutoff_freq_manual, inpaint_length],
    #         label="Hızlı Ayarlar"
    #     )
    
    restore_btn.click(
        fn=restore_audio,
        inputs=[audio_input, mode, n_steps, cutoff_freq_auto, cutoff_freq_manual, inpaint_length],
        outputs=[audio_output, info_output]
    )
    
    gr.Markdown("""
    ---
    ### 📖 Kaynak
    - 📄 [Paper](https://arxiv.org/abs/2501.11311)
    - 💻 [GitHub](https://github.com/test4373/diffusion-audio-restoration-colab-Kaggle-.git)
    - 🤗 [Models](https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge)
    """)

if __name__ == "__main__":
    # Komut satırı argümanlar��
    parser = argparse.ArgumentParser(description='A2SB Gradio App')
    parser.add_argument('--share', action='store_true', help='Create public URL')
    parser.add_argument('--port', type=int, default=7860, help='Port number')
    parser.add_argument('--ngrok', action='store_true', help='Use ngrok tunnel (Kaggle)')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🎵 A2SB Audio Restoration")
    print("="*60)
    
    # Ngrok setup for Kaggle
    ngrok_tunnel = None
    if IN_KAGGLE or args.ngrok:
        try:
            from pyngrok import ngrok as pyngrok_module
            
            print("🌐 Kaggle ortamı tespit edildi")
            print("📁 Çıktılar: /kaggle/working/gradio_outputs/")
            
            # Ngrok token kontrolü
            ngrok_token = os.environ.get('NGROK_TOKEN', '')
            if ngrok_token:
                print("🔑 Ngrok token bulundu, ayarlanıyor...")
                pyngrok_module.set_auth_token(ngrok_token)
                print("✅ Ngrok token ayarlandı")
            else:
                print("⚠️ Ngrok token bulunamadı")
                print("💡 Token için: https://dashboard.ngrok.com/get-started/your-authtoken")
                print("📝 Token'ı notebook'ta NGROK_TOKEN environment variable olarak ayarlayın")
            
            print("🔗 Ngrok tunnel oluşturuluyor...")
            
            # Ngrok tunnel oluştur
            ngrok_tunnel = pyngrok_module.connect(args.port)
            public_url = ngrok_tunnel.public_url
            
            print(f"\n✅ Ngrok tunnel oluşturuldu!")
            print(f"🔗 Public URL: {public_url}")
            print(f"📝 Bu URL'yi tarayıcınızda açın\n")
            
            share = True  # Ngrok ile birlikte share=True kullan (Kaggle için gerekli)
        except ImportError:
            print("⚠️ pyngrok yüklü değil, Gradio share kullanılacak")
            print("💡 Ngrok için: pip install pyngrok")
            share = True
        except Exception as e:
            print(f"⚠️ Ngrok hatası: {e}")
            print("📝 Gradio share kullanılacak")
            share = True
    else:
        print("💻 Yerel ortam")
        share = args.share
    
    print("\n🚀 Gradio başlatılıyor...")
    print("="*60 + "\n")
    
    try:
        demo.launch(
            share=share,
            server_port=args.port,
            debug=True,
            show_error=True,
            show_api=False,  # API dokümantasyonunu devre dışı bırak (JSON schema hatası önleme)
            server_name="0.0.0.0" if IN_KAGGLE else "127.0.0.1"
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Uygulama kapatılıyor...")
        if ngrok_tunnel:
            pyngrok_module.disconnect(ngrok_tunnel.public_url)
            print("✅ Ngrok tunnel kapatıldı")
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        if ngrok_tunnel:
            pyngrok_module.disconnect(ngrok_tunnel.public_url)
