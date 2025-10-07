# 🎵 A2SB: Audio-to-Audio Schrödinger Bridge

[![License](https://img.shields.io/badge/License-NVIDIA-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app/)

**High-Quality Audio Restoration using Diffusion Models**

Restore degraded audio to high-quality 44.1kHz music using NVIDIA's state-of-the-art A2SB model. This repository includes an optimized Gradio web interface for easy use!

## 🌟 Features

- ✅ **44.1kHz High-Resolution** music restoration
- ✅ **Bandwidth Extension** - Restore high frequencies from low-quality audio
- ✅ **Audio Inpainting** - Fill in missing audio segments
- ✅ **Long Audio Support** - Process hours of audio
- ✅ **End-to-End** - No vocoder required
- ✅ **User-Friendly Gradio Interface** - Drag-and-drop simplicity
- ✅ **GPU Memory Optimized** - Works on 8GB+ GPUs
- ✅ **Google Colab Ready** - Complete notebook included

## 🚀 Quick Start

### Option 1: Google Colab (Recommended for Beginners)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1m-G7gYXtumpqIVz2O6hCwLhLTWFMM6nw?usp=sharing)

1. Click the badge above
2. Run all cells in order
3. Use the Gradio interface that appears!

### Option 2: Kaggle (Free GPU Alternative)

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/YOUR_USERNAME/diffusion-audio-restoration/blob/main/A2SB_Complete_Kaggle.ipynb)

1. Click the badge above
2. Enable GPU in Settings (P100 recommended)
3. Run all cells in order
4. Use the Kaggle-optimized Gradio interface!

**Why Kaggle?**
- ✅ Free GPU access (30 hours/week)
- ✅ P100 GPU with 16GB VRAM
- ✅ 30GB RAM, 73GB disk
- ✅ No subscription required
- ✅ Kaggle-optimized Gradio interface (`gradio_app_kaggle.py`)

### Option 3: Local Installation

#### Prerequisites
- Python 3.10+
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.8 or higher

#### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/diffusion-audio-restoration.git
cd diffusion-audio-restoration

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install lightning "pytorch-lightning>=2.0.0"
pip install numpy scipy matplotlib librosa soundfile
pip install einops gradio "jsonargparse[signatures]>=4.0.0"
pip install nest-asyncio

# Optional but recommended
pip install rotary-embedding-torch
pip install ssr-eval
```

#### Download Models

```bash
# Create checkpoint directory
mkdir -p ckpt

# Download models (each ~1.5GB)
wget -O ckpt/A2SB_onesplit_0.0_1.0_release.ckpt \
  https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge/resolve/main/ckpt/A2SB_onesplit_0.0_1.0_release.ckpt

wget -O ckpt/A2SB_twosplit_0.5_1.0_release.ckpt \
  https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge/resolve/main/ckpt/A2SB_twosplit_0.5_1.0_release.ckpt
```

#### Update Configuration

```bash
# Update model paths in config
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

#### Launch Gradio Interface

```bash
python gradio_app.py
```

The interface will open in your browser at `http://localhost:7860`

## 📖 Usage Guide

### Gradio Web Interface

1. **Upload Audio**: Drag and drop your audio file or record from microphone
2. **Choose Mode**:
   - **Bandwidth Extension**: Restore high frequencies (for low-quality MP3s)
   - **Inpainting**: Fill in missing audio segments
3. **Adjust Settings** (optional):
   - **Sampling Steps**: 25-100 (higher = better quality, slower)
   - **Cutoff Frequency**: Auto-detect or manual (for bandwidth extension)
   - **Inpainting Length**: 0.1-1.0 seconds (for inpainting)
4. **Click "🚀 Restore"** and wait for processing
5. **Listen & Download** the restored audio

### Command Line Interface

```bash
# Bandwidth extension
python ensembled_inference_api.py predict \
  -c configs/ensemble_2split_sampling.yaml \
  -c configs/inference_files_upsampling.yaml \
  --model.predict_n_steps=50 \
  --model.output_audio_filename=output.wav

# Audio inpainting
python ensembled_inference_api.py predict \
  -c configs/ensemble_2split_sampling.yaml \
  -c configs/inference_files_inpainting.yaml \
  --model.predict_n_steps=50 \
  --model.output_audio_filename=output.wav
```

## ⚙️ Configuration

### Quality Settings

| Setting | Fast | Balanced | Best |
|---------|------|----------|------|
| Sampling Steps | 25-30 | 50-75 | 75-100 |
| Processing Time (10s audio) | ~1-2 min | ~2-3 min | ~4-5 min |
| Quality | Good | Excellent | Outstanding |

### GPU Memory Optimization

The code includes several optimizations for limited GPU memory:

- **Mixed Precision (FP16)**: ~50% memory reduction
- **Batch Size = 1**: Minimal memory footprint
- **Segment Length Reduction**: Process audio in smaller chunks
- **Automatic GPU Cleanup**: Clears memory before inference

For 8GB GPUs, use:
- Sampling steps: 25-50
- Audio length: Up to 30 seconds at a time

For 16GB+ GPUs:
- Sampling steps: 50-100
- Audio length: Up to several minutes

## 🔧 Troubleshooting

### CUDA Out of Memory

```bash
# Clear GPU memory
python kill_gpu_processes.py

# Or manually
nvidia-smi
# Find process ID and kill it
kill -9 <PID>
```

**Solutions:**
1. Reduce sampling steps to 25-30
2. Split long audio into shorter segments
3. Close other GPU applications
4. Restart Python kernel/runtime

### Model Not Found

```bash
# Verify models are downloaded
ls -lh ckpt/
# Should show two .ckpt files (~1.5GB each)

# Re-download if needed
rm -rf ckpt/*.ckpt
# Run download commands again
```

### Audio Format Issues

```python
# Convert any audio to WAV
import librosa
import soundfile as sf

y, sr = librosa.load('input.mp3', sr=44100)
sf.write('input.wav', y, sr)
```

## 📊 Performance Benchmarks

| GPU | Audio Length | Steps | Time | Memory |
|-----|--------------|-------|------|--------|
| T4 (16GB) | 10s | 50 | ~2-3 min | ~6GB |
| T4 (16GB) | 30s | 50 | ~5-7 min | ~8GB |
| V100 (32GB) | 60s | 75 | ~8-10 min | ~12GB |
| A100 (40GB) | 120s | 100 | ~15-20 min | ~18GB |

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{kong2025a2sb,
  title={A2SB: Audio-to-Audio Schrodinger Bridges},
  author={Kong, Zhifeng and Shih, Kevin J and Nie, Weili and Vahdat, Arash and Lee, Sang-gil and Santos, Joao Felipe and Jukic, Ante and Valle, Rafael and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:2501.11311},
  year={2025}
}
```

## 📄 License

- **Model**: NVIDIA OneWay NonCommercial License
- **Code**: NVIDIA Source Code License - Non Commercial

See [LICENSE](LICENSE) for details.

**For commercial use**, please contact NVIDIA.

## 🔗 Resources

- 📄 **Paper**: [arXiv:2501.11311](https://arxiv.org/abs/2501.11311)
- 💻 **Original GitHub**: [NVIDIA/diffusion-audio-restoration](https://github.com/NVIDIA/diffusion-audio-restoration)
- 🎬 **Demo**: [NVIDIA Research](https://research.nvidia.com/labs/adlr/A2SB/)
- 🤗 **Models**: [HuggingFace](https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge)

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/diffusion-audio-restoration/issues)
- **Discussions**: [GitHub Discussions](https://github.com/YOUR_USERNAME/diffusion-audio-restoration/discussions)

## 🙏 Acknowledgments

This project is based on NVIDIA's A2SB research. Special thanks to:
- NVIDIA Research Team
- Original paper authors
- Open-source community

## ⭐ Star History

If you find this project useful, please consider starring the repository!

---

**Made with ❤️ by the community**

**Optimized for ease of use with Gradio interface and GPU memory management**
