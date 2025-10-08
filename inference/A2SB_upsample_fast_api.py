# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for A2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
# FAST INFERENCE VERSION - Optimized for speed
# ---------------------------------------------------------------

import os
import numpy as np 
import json
import argparse
import glob
from subprocess import Popen, PIPE
import yaml
import time 
from datetime import datetime
import shutil
import csv
from tqdm import tqdm

import librosa
import soundfile as sf


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def save_yaml(data, prefix="../configs/temp"):
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rnd_num = np.random.rand()
    rnd_num = rnd_num - rnd_num % 0.000001
    file_name = f"{prefix}_{timestamp}_{rnd_num}.yaml"
    with open(file_name, 'w') as f:
        yaml.dump(data, f)
    return file_name


def shell_run_cmd(cmd):
    print('running:', cmd)
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    stdout, stderr = p.communicate()
    print(stdout)
    print(stderr)


def compute_rolloff_freq(audio_file, roll_percent=0.99):
    y, sr = librosa.load(audio_file, sr=None)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=roll_percent)[0]
    rolloff = int(np.mean(rolloff))
    print('99 percent rolloff:', rolloff)
    return rolloff


def upsample_one_sample_fast(
    audio_filename, 
    output_audio_filename, 
    predict_n_steps=50,
    use_compile=True,
    precision="fp16",
    compile_mode="reduce-overhead",
    use_cudnn_benchmark=True,
    use_tf32=True
):
    """
    Hızlandırılmış audio upsampling
    
    Args:
        audio_filename: Input audio dosyası
        output_audio_filename: Output audio dosyası
        predict_n_steps: Sampling step sayısı (daha az = daha hızlı)
        use_compile: torch.compile() kullan (PyTorch 2.0+)
        precision: "fp16", "bf16", veya "fp32"
        compile_mode: "default", "reduce-overhead", veya "max-autotune"
        use_cudnn_benchmark: cuDNN benchmark modu
        use_tf32: TF32 kullan (Ampere+ GPU'lar için)
    """
    
    assert output_audio_filename != audio_filename, "output filename cannot be input filename"

    inference_config = load_yaml('../configs/inference_files_upsampling.yaml')
    inference_config['data']['predict_filelist'] = [{
        'filepath': audio_filename,
        'output_subdir': '.'
    }]

    cutoff_freq = compute_rolloff_freq(audio_filename, roll_percent=0.99)
    inference_config['data']['transforms_aug'][0]['init_args']['upsample_mask_kwargs'] = {
        'min_cutoff_freq': cutoff_freq,
        'max_cutoff_freq': cutoff_freq
    }
    temporary_yaml_file = save_yaml(inference_config)

    # Hızlı inference parametreleri
    fast_params = [
        f"--model.use_fast_inference=True",
        f"--model.use_compile={use_compile}",
        f"--model.precision={precision}",
        f"--model.compile_mode={compile_mode}",
        f"--model.use_cudnn_benchmark={use_cudnn_benchmark}",
        f"--model.use_tf32={use_tf32}"
    ]
    
    fast_params_str = " ".join(fast_params)

    cmd = f"cd ../; \
        python ensembled_inference_fast_api.py predict \
            -c configs/ensemble_2split_sampling.yaml \
            -c {temporary_yaml_file.replace('../', '')} \
            --model.predict_n_steps={predict_n_steps} \
            --model.output_audio_filename={output_audio_filename} \
            {fast_params_str}; \
        cd inference/"
        
    print("\n" + "="*60)
    print("🚀 FAST INFERENCE MODE")
    print("="*60)
    print(f"Compile: {use_compile}")
    print(f"Precision: {precision}")
    print(f"Compile mode: {compile_mode}")
    print(f"cuDNN benchmark: {use_cudnn_benchmark}")
    print(f"TF32: {use_tf32}")
    print(f"Steps: {predict_n_steps}")
    print("="*60 + "\n")
    
    start_time = time.time()
    shell_run_cmd(cmd)
    end_time = time.time()
    
    print("\n" + "="*60)
    print(f"✓ Processing completed in {end_time - start_time:.2f} seconds")
    print("="*60 + "\n")
    
    os.remove(temporary_yaml_file)


def benchmark_inference(
    audio_filename,
    output_audio_filename,
    predict_n_steps=50,
    num_runs=3
):
    """
    Farklı konfigürasyonları benchmark et
    """
    configs = [
        {
            "name": "FP32 (Baseline)",
            "precision": "fp32",
            "use_compile": False
        },
        {
            "name": "FP16 + Compile",
            "precision": "fp16",
            "use_compile": True,
            "compile_mode": "reduce-overhead"
        },
        {
            "name": "FP16 + Max Autotune",
            "precision": "fp16",
            "use_compile": True,
            "compile_mode": "max-autotune"
        }
    ]
    
    results = []
    
    print("\n" + "="*60)
    print("BENCHMARK MODE")
    print("="*60)
    print(f"Audio: {audio_filename}")
    print(f"Steps: {predict_n_steps}")
    print(f"Runs per config: {num_runs}")
    print("="*60 + "\n")
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        print("-" * 60)
        
        times = []
        for run in range(num_runs):
            print(f"Run {run+1}/{num_runs}...")
            
            start_time = time.time()
            upsample_one_sample_fast(
                audio_filename,
                output_audio_filename,
                predict_n_steps=predict_n_steps,
                use_compile=config.get("use_compile", False),
                precision=config.get("precision", "fp32"),
                compile_mode=config.get("compile_mode", "reduce-overhead")
            )
            end_time = time.time()
            
            elapsed = end_time - start_time
            times.append(elapsed)
            print(f"Time: {elapsed:.2f}s")
            
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        results.append({
            "config": config["name"],
            "avg_time": avg_time,
            "std_time": std_time,
            "times": times
        })
        
        print(f"\nAverage: {avg_time:.2f}s ± {std_time:.2f}s")
        print("-" * 60)
    
    # Sonuçları yazdır
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    baseline_time = results[0]["avg_time"]
    
    for result in results:
        speedup = baseline_time / result["avg_time"]
        print(f"\n{result['config']}:")
        print(f"  Average time: {result['avg_time']:.2f}s ± {result['std_time']:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")
        
    print("\n" + "="*60 + "\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Fast Audio Upsampling with PyTorch Optimizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (fast mode)
  python A2SB_upsample_fast_api.py -f input.wav -o output.wav
  
  # Custom settings
  python A2SB_upsample_fast_api.py -f input.wav -o output.wav \\
      --predict_n_steps 30 \\
      --precision fp16 \\
      --compile_mode max-autotune
  
  # Benchmark mode
  python A2SB_upsample_fast_api.py -f input.wav -o output.wav --benchmark
        """
    )
    
    parser.add_argument('-f', '--audio_filename', type=str, 
                       help='audio filename to be upsampled', required=True)
    parser.add_argument('-o', '--output_audio_filename', type=str, 
                       help='path to save upsampled audio', required=True)
    parser.add_argument('-n', '--predict_n_steps', type=int, 
                       help='number of sampling steps (default: 50, lower=faster)', 
                       default=50)
    
    # Fast inference parametreleri
    parser.add_argument('--use_compile', type=bool, default=True,
                       help='use torch.compile() (PyTorch 2.0+)')
    parser.add_argument('--precision', type=str, default='fp16',
                       choices=['fp16', 'bf16', 'fp32'],
                       help='precision mode (fp16=fastest, fp32=most accurate)')
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead',
                       choices=['default', 'reduce-overhead', 'max-autotune'],
                       help='torch.compile() mode')
    parser.add_argument('--use_cudnn_benchmark', type=bool, default=True,
                       help='enable cuDNN benchmark mode')
    parser.add_argument('--use_tf32', type=bool, default=True,
                       help='enable TF32 (Ampere+ GPUs)')
    
    # Benchmark modu
    parser.add_argument('--benchmark', action='store_true',
                       help='run benchmark with different configurations')
    parser.add_argument('--benchmark_runs', type=int, default=3,
                       help='number of runs per configuration in benchmark mode')
    
    args = parser.parse_args()

    if args.benchmark:
        benchmark_inference(
            args.audio_filename,
            args.output_audio_filename,
            args.predict_n_steps,
            args.benchmark_runs
        )
    else:
        upsample_one_sample_fast(
            audio_filename=args.audio_filename,
            output_audio_filename=args.output_audio_filename,
            predict_n_steps=args.predict_n_steps,
            use_compile=args.use_compile,
            precision=args.precision,
            compile_mode=args.compile_mode,
            use_cudnn_benchmark=args.use_cudnn_benchmark,
            use_tf32=args.use_tf32
        )


if __name__ == '__main__':
    main()
