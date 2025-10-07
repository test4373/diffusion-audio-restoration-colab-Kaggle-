# 🚀 Quick Setup Guide

## For GitHub Users

### Step 1: Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/diffusion-audio-restoration.git
cd diffusion-audio-restoration
```

### Step 2: Install Dependencies
```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

Or install manually:
```bash
pip install lightning "pytorch-lightning>=2.0.0"
pip install numpy scipy matplotlib librosa soundfile
pip install einops gradio "jsonargparse[signatures]>=4.0.0"
pip install nest-asyncio rotary-embedding-torch
```

### Step 3: Download Models
```bash
mkdir -p ckpt

# Model 1 (~1.5GB)
wget -O ckpt/A2SB_onesplit_0.0_1.0_release.ckpt \
  https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge/resolve/main/ckpt/A2SB_onesplit_0.0_1.0_release.ckpt

# Model 2 (~1.5GB)
wget -O ckpt/A2SB_twosplit_0.5_1.0_release.ckpt \
  https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge/resolve/main/ckpt/A2SB_twosplit_0.5_1.0_release.ckpt
```

### Step 4: Update Configuration
```bash
python -c "
import yaml
with open('configs/ensemble_2split_sampling.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['model']['pretrained_checkpoints'] = [
    'ckpt/A2SB_onesplit_0.0_1.0_release.ckpt',
    'ckpt/A2SB_twosplit_0.5_1.0_release.ckpt'
]
with open('configs/ensemble_2split_sampling.yaml', 'w') as f:
    yaml.dump(config, f)
print('✓ Configuration updated')
"
```

### Step 5: Launch Gradio
```bash
python gradio_app.py
```

Open your browser at `http://localhost:7860`

---

## For Google Colab Users

### Option 1: Use the Notebook
1. Open `A2SB_Complete_Colab.ipynb` in Google Colab
2. Run all cells in order
3. Use the Gradio interface!

### Option 2: Quick Commands
```python
# In Colab cell 1: Clone and setup
!git clone https://github.com/YOUR_USERNAME/diffusion-audio-restoration.git
%cd diffusion-audio-restoration
!pip install -q torch torchvision torchaudio
!pip install -q lightning numpy scipy librosa soundfile einops gradio nest-asyncio

# In Colab cell 2: Download models
!mkdir -p ckpt
!wget -O ckpt/A2SB_onesplit_0.0_1.0_release.ckpt https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge/resolve/main/ckpt/A2SB_onesplit_0.0_1.0_release.ckpt
!wget -O ckpt/A2SB_twosplit_0.5_1.0_release.ckpt https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge/resolve/main/ckpt/A2SB_twosplit_0.5_1.0_release.ckpt

# In Colab cell 3: Update config and launch
import yaml
with open('configs/ensemble_2split_sampling.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['model']['pretrained_checkpoints'] = [
    'ckpt/A2SB_onesplit_0.0_1.0_release.ckpt',
    'ckpt/A2SB_twosplit_0.5_1.0_release.ckpt'
]
with open('configs/ensemble_2split_sampling.yaml', 'w') as f:
    yaml.dump(config, f)

!python gradio_app.py
```

---

## Troubleshooting

### GPU Memory Issues
```bash
# Clear GPU memory
python kill_gpu_processes.py

# Or check GPU usage
nvidia-smi
```

### Model Download Issues
If wget doesn't work, use Python:
```python
import urllib.request
import os

os.makedirs('ckpt', exist_ok=True)

urls = {
    'ckpt/A2SB_onesplit_0.0_1.0_release.ckpt': 
        'https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge/resolve/main/ckpt/A2SB_onesplit_0.0_1.0_release.ckpt',
    'ckpt/A2SB_twosplit_0.5_1.0_release.ckpt': 
        'https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge/resolve/main/ckpt/A2SB_twosplit_0.5_1.0_release.ckpt'
}

for path, url in urls.items():
    if not os.path.exists(path):
        print(f'Downloading {path}...')
        urllib.request.urlretrieve(url, path)
        print(f'✓ Downloaded {path}')
```

### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade --force-reinstall torch torchvision torchaudio
pip install --upgrade lightning gradio
```

---

## System Requirements

### Minimum
- Python 3.10+
- NVIDIA GPU with 8GB VRAM
- CUDA 11.8+
- 16GB RAM
- 10GB disk space (for models)

### Recommended
- Python 3.10+
- NVIDIA GPU with 16GB+ VRAM
- CUDA 12.0+
- 32GB RAM
- 20GB disk space

---

## Quick Test

After setup, test with a sample audio:
```bash
# Create a test audio file
python -c "
import numpy as np
import soundfile as sf
sr = 44100
duration = 5
t = np.linspace(0, duration, sr * duration)
audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
sf.write('test_input.wav', audio, sr)
print('✓ Test audio created')
"

# Run inference
python ensembled_inference_api.py predict \
  -c configs/ensemble_2split_sampling.yaml \
  -c configs/inference_files_upsampling.yaml \
  --model.predict_n_steps=25 \
  --model.output_audio_filename=test_output.wav
```

---

## Need Help?

- 📖 Read the [README.md](README.md)
- 🐛 Report issues on [GitHub Issues](https://github.com/YOUR_USERNAME/diffusion-audio-restoration/issues)
- 💬 Ask questions in [Discussions](https://github.com/YOUR_USERNAME/diffusion-audio-restoration/discussions)

---

**Happy Restoring! 🎵**
