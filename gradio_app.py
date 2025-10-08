import gradio as gr
import os
import sys
import yaml
import librosa
import numpy as np
from datetime import datetime

# ============================================================================
# WORKING DIRECTORY AND PYTHON PATH SETTINGS
# ============================================================================

# Set working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
os.chdir(SCRIPT_DIR)

# Add to Python path - This is very important!
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

print(f"📁 Working Directory: {SCRIPT_DIR}")
print(f"🐍 Python Path: {sys.path[:3]}")

OUTPUT_DIR = "gradio_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# IMPORT MODULES
# ============================================================================

try:
    from A2SB_lightning_module_api import TimePartitionedPretrainedSTFTBridgeModel
    from datasets.datamodule import STFTAudioDataModule
    print("✅ Modules loaded successfully")
except ImportError as e:
    print(f"❌ Module loading error: {e}")
    print("Please make sure you are in the correct directory")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_rolloff_freq(audio_file, roll_percent=0.99):
    """Automatic cutoff frequency detection"""
    try:
        y, sr = librosa.load(audio_file, sr=None)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=roll_percent)[0]
        return int(np.mean(rolloff))
    except Exception as e:
        print(f"Rolloff calculation error: {e}")
        return 2000

def kill_gpu_processes():
    """Kill other Python processes using GPU"""
    try:
        import subprocess
        
        # Find GPU processes using nvidia-smi
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
                        # Kill process
                        if sys.platform == 'win32':
                            subprocess.run(['taskkill', '/F', '/PID', pid], 
                                         capture_output=True, check=False)
                        else:
                            subprocess.run(['kill', '-9', pid], 
                                         capture_output=True, check=False)
                        print(f"❌ GPU process killed (PID: {pid})")
                except:
                    pass
    except:
        pass

def clear_gpu_memory():
    """Clear GPU memory"""
    try:
        import gc
        import torch
        
        # First clear other processes
        kill_gpu_processes()
        
        # Python garbage collector
        gc.collect()
        
        if torch.cuda.is_available():
            # Clear all GPUs
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            
            torch.cuda.synchronize()
            print("🧹 GPU memory cleared")
            return True
        return False
    except Exception as e:
        print(f"⚠️ GPU memory clearing error: {e}")
        return False

def run_inference_direct(config_files, n_steps, output_path):
    """Run inference via CLI - using subprocess"""
    try:
        import subprocess
        
        # Clear GPU memory
        clear_gpu_memory()
        
        # Set PYTHONPATH as environment variable
        env = os.environ.copy()
        env['PYTHONPATH'] = SCRIPT_DIR
        
        # CUDA memory optimizations
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
        env['CUDA_LAUNCH_BLOCKING'] = '0'
        
        # Special settings for Colab
        try:
            import google.colab
            IN_COLAB = True
        except:
            IN_COLAB = False
        
        if IN_COLAB:
            env['COLAB_GPU'] = '1'

        # NCCL/IPC settings (for Colab security)
        env.setdefault('NCCL_P2P_DISABLE', '1')
        env.setdefault('NCCL_SHM_DISABLE', '1')
        env.setdefault('NCCL_IB_DISABLE', '1')

        # Read predict_filelist length from config (determine behavior for single/multi file)
        num_predict_items = 1
        try:
            with open(config_files[1], 'r') as f:
                cfg_tmp = yaml.safe_load(f)
            num_predict_items = len(cfg_tmp.get('data', {}).get('predict_filelist', [])) or 1
        except Exception:
            pass

        # Detect GPU count and select appropriate strategy
        try:
            import torch
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            num_gpus = 0

        predict_bs = 8  # Safer default for memory

        if num_gpus >= 2:
            try:
                import torch
                free_per_gpu = []
                for idx in range(num_gpus):
                    try:
                        with torch.cuda.device(idx):
                            free, total = torch.cuda.mem_get_info()
                        free_per_gpu.append((idx, free))
                    except Exception:
                        free_per_gpu.append((idx, 0))
                # Sort by most free memory
                free_per_gpu.sort(key=lambda x: x[1], reverse=True)

                if num_predict_items >= 2:
                    # Use multi-GPU (DDP) for multiple files
                    chosen = [str(free_per_gpu[i][0]) for i in range(min(2, len(free_per_gpu)))]
                    env['CUDA_VISIBLE_DEVICES'] = ','.join(chosen)
                    env.setdefault('MASTER_ADDR', '127.0.0.1')
                    env.setdefault('MASTER_PORT', '12975')
                    trainer_args = [
                        '--trainer.strategy=ddp',
                        f"--trainer.devices={len(chosen)}",
                        '--trainer.accelerator=gpu',
                        '--trainer.precision=16-mixed'
                    ]
                    print(f"🚀 Multi-GPU DDP enabled: CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} | items={num_predict_items}")
                else:
                    # For single file prediction, select the GPU with most free memory
                    best_gpu = str(free_per_gpu[0][0])
                    env['CUDA_VISIBLE_DEVICES'] = best_gpu
                    trainer_args = [
                        '--trainer.strategy=auto',
                        '--trainer.devices=1',
                        '--trainer.accelerator=gpu',
                        '--trainer.precision=16-mixed'
                    ]
                    print(f"🎯 Single file: selected GPU with most free memory -> CUDA_VISIBLE_DEVICES={best_gpu}")
            except Exception:
                # Fallback: both GPUs visible, but use single GPU
                env['CUDA_VISIBLE_DEVICES'] = '0,1'
                trainer_args = [
                    '--trainer.strategy=auto',
                    '--trainer.devices=1',
                    '--trainer.accelerator=gpu',
                    '--trainer.precision=16-mixed'
                ]
                print("⚠️ Error during GPU selection: single GPU fallback")
        elif num_gpus == 1:
            env['CUDA_VISIBLE_DEVICES'] = '0'
            trainer_args = [
                '--trainer.strategy=auto',
                '--trainer.devices=1',
                '--trainer.accelerator=gpu',
                '--trainer.precision=16-mixed'
            ]
            print("🎯 Single GPU detected: CUDA_VISIBLE_DEVICES=0")
        else:
            trainer_args = [
                '--trainer.strategy=auto',
                '--trainer.devices=1',
                '--trainer.accelerator=cpu'
            ]
            print("🖥️ No GPU detected: CPU mode")

        # Run Python with -u (unbuffered) flag
        cmd = [
            sys.executable,
            '-u',  # Unbuffered output
            os.path.join(SCRIPT_DIR, 'ensembled_inference_api.py'),
            'predict',
            '-c', config_files[0],
            '-c', config_files[1],
            f'--model.predict_n_steps={n_steps}',
            f'--model.output_audio_filename={output_path}',
            f'--model.predict_batch_size={predict_bs}',
        ] + trainer_args + [
            '--data.batch_size=1'  # Reduce batch size to 1
        ]
        
        print(f"🔧 Running command...")
        print(f"📁 PYTHONPATH: {env['PYTHONPATH']}")
        print(f"📁 Working directory: {SCRIPT_DIR}")
        print(f"📄 Config 1: {config_files[0]}")
        print(f"📄 Config 2: {config_files[1]}")
        print(f"📤 Output: {output_path}")
        print(f"💻 Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=SCRIPT_DIR,
            env=env
        )
        
        if result.returncode == 0:
            print(f"✅ Inference completed!")
            return True, None
        else:
            # Show more error details
            error_msg = f"Error code: {result.returncode}\n\n"
            error_msg += f"STDOUT (last 2000 chars):\n{result.stdout[-2000:]}\n\n"
            error_msg += f"STDERR (last 2000 chars):\n{result.stderr[-2000:]}"
            print(f"❌ {error_msg}")
            return False, error_msg
        
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        return False, error_msg

def restore_audio(audio_file, mode, n_steps, cutoff_freq_auto, cutoff_freq_manual, 
                  inpaint_length, progress=gr.Progress()):
    """Main audio restoration function"""
    try:
        progress(0, desc="🚀 Starting...")
        
        if audio_file is None:
            return None, "❌ Please upload an audio file!"
        
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_filename = os.path.basename(audio_file)
        base_name = input_filename.rsplit('.', 1)[0] if '.' in input_filename else input_filename
        output_filename = f"{timestamp}_{mode}_{base_name}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        progress(0.1, desc="📊 Analyzing audio file...")
        
        # Get audio info
        y, sr = librosa.load(audio_file, sr=None)
        duration = len(y) / sr
        
        info_text = f"## 📊 Audio Information\n\n"
        info_text += f"- **Sampling Rate:** {sr} Hz\n"
        info_text += f"- **Duration:** {duration:.2f} seconds\n"
        info_text += f"- **Samples:** {len(y):,}\n\n"
        
        # Prepare config files
        base_config = os.path.join(SCRIPT_DIR, 'configs', 'ensemble_2split_sampling.yaml')
        
        if mode == "bandwidth":
            progress(0.2, desc="🎯 Preparing bandwidth extension...")
            
            # Determine cutoff frequency
            if cutoff_freq_auto:
                cutoff_freq = compute_rolloff_freq(audio_file)
                info_text += f"🎯 **Auto Cutoff:** {cutoff_freq} Hz\n"
            else:
                cutoff_freq = int(cutoff_freq_manual)
                info_text += f"🎯 **Manual Cutoff:** {cutoff_freq} Hz\n"
            
            info_text += f"⚙️ **Sampling Steps:** {n_steps}\n\n"
            
            # Prepare config
            config_path = os.path.join(SCRIPT_DIR, 'configs', 'inference_files_upsampling.yaml')
            if not os.path.exists(config_path):
                return None, f"❌ Config file not found: {config_path}"
            
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
            
            # Memory optimization: reduce segment_length
            # Original: 130560 samples (~3 seconds @ 44.1kHz)
            # New: 65280 samples (~1.5 seconds @ 44.1kHz)
            if 'segment_length' in config['data']:
                config['data']['segment_length'] = 65280
                info_text += f"⚙️ **Segment Length:** 65280 samples (~1.5s) - Memory optimization\n"
            
            temp_config = os.path.join(SCRIPT_DIR, 'configs', f'temp_gradio_{timestamp}.yaml')
            with open(temp_config, 'w') as f:
                yaml.dump(config, f)
        
        elif mode == "inpainting":
            progress(0.2, desc="🎨 Preparing audio inpainting...")
            
            info_text += f"🎯 **Inpainting Length:** {inpaint_length}s\n"
            info_text += f"⚙️ **Sampling Steps:** {n_steps}\n\n"
            
            # Prepare config
            config_path = os.path.join(SCRIPT_DIR, 'configs', 'inference_files_inpainting.yaml')
            if not os.path.exists(config_path):
                return None, f"❌ Config file not found: {config_path}"
            
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
        
        progress(0.3, desc="🔄 Running model... (This may take a few minutes)")
        
        # Run inference
        success, error = run_inference_direct([base_config, temp_config], n_steps, output_path)
        
        # Delete temporary config
        if os.path.exists(temp_config):
            os.remove(temp_config)
        
        if not success:
            return None, f"## ❌ Error\n\n```\n{error}\n```"
        
        progress(0.9, desc="✨ Finalizing...")
        
        # Check output file
        if os.path.exists(output_path):
            info_text += f"---\n\n## ✅ Processing Complete!\n\n"
            info_text += f"���� **Output File:** `{output_filename}`\n\n"
            
            # Output audio info
            y_out, sr_out = librosa.load(output_path, sr=None)
            info_text += f"## 📊 Restored Audio\n\n"
            info_text += f"- **Sampling Rate:** {sr_out} Hz\n"
            info_text += f"- **Duration:** {len(y_out)/sr_out:.2f} seconds\n"
            info_text += f"- **Samples:** {len(y_out):,}\n\n"
            
            # Spectral features
            cent_orig = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
            cent_rest = np.mean(librosa.feature.spectral_centroid(y=y_out, sr=sr_out)[0])
            rolloff_orig = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99)[0])
            rolloff_rest = np.mean(librosa.feature.spectral_rolloff(y=y_out, sr=sr_out, roll_percent=0.99)[0])
            
            info_text += f"## 📈 Spectral Analysis\n\n"
            info_text += f"| Feature | Original | Restored | Change |\n"
            info_text += f"|---------|----------|----------|---------|\\n"
            info_text += f"| Spectral Centroid | {cent_orig:.0f} Hz | {cent_rest:.0f} Hz | {((cent_rest-cent_orig)/cent_orig*100):+.1f}% |\n"
            info_text += f"| Spectral Rolloff | {rolloff_orig:.0f} Hz | {rolloff_rest:.0f} Hz | {((rolloff_rest-rolloff_orig)/rolloff_orig*100):+.1f}% |\n"
            
            progress(1.0, desc="✅ Complete!")
            return output_path, info_text
        else:
            return None, info_text + "\n---\n\n## ❌ Error\n\nOutput file could not be created."
    
    except Exception as e:
        import traceback
        error_msg = f"## ❌ Error Occurred\n\n```\n{str(e)}\n\n{traceback.format_exc()}\n```"
        return None, error_msg

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

with gr.Blocks(title="A2SB Audio Restoration", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎵 A2SB: Audio-to-Audio Schrödinger Bridge
    ### High-Quality Audio Restoration - NVIDIA
    
    Restore your audio files with AI!
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Input")
            
            audio_input = gr.Audio(
                label="Upload Audio File",
                type="filepath",
                sources=["upload", "microphone"]
            )
            
            mode = gr.Radio(
                choices=["bandwidth", "inpainting"],
                value="bandwidth",
                label="Restoration Mode"
            )
            
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                n_steps = gr.Slider(25, 100, 50, step=5, label="Sampling Steps")
                cutoff_freq_auto = gr.Checkbox(True, label="Auto Cutoff")
                cutoff_freq_manual = gr.Slider(1000, 10000, 2000, step=100, label="Manual Cutoff (Hz)", visible=False)
                inpaint_length = gr.Slider(0.1, 1.0, 0.3, step=0.1, label="Inpainting Length (s)")
            
            cutoff_freq_auto.change(
                fn=lambda x: gr.update(visible=not x),
                inputs=[cutoff_freq_auto],
                outputs=[cutoff_freq_manual]
            )
            
            restore_btn = gr.Button("🚀 Restore", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Output")
            audio_output = gr.Audio(label="Restored Audio", type="filepath")
            info_output = gr.Markdown("Upload an audio file and click 'Restore'.")
    
    restore_btn.click(
        fn=restore_audio,
        inputs=[audio_input, mode, n_steps, cutoff_freq_auto, cutoff_freq_manual, inpaint_length],
        outputs=[audio_output, info_output]
    )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎵 A2SB Audio Restoration")
    print("="*60)
    
    # Check for Colab
    try:
        import google.colab
        IN_COLAB = True
        print("🌐 Google Colab environment")
    except:
        IN_COLAB = False
        print("💻 Local environment")
    
    print("\n🚀 Launching Gradio...")
    print("="*60 + "\n")
    
    demo.launch(share=IN_COLAB, debug=True, show_error=True)
