# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for A2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
# FAST INFERENCE VERSION - Optimized for speed
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule, Callback
from utils import SequenceLength, average_key_value
from typing import Optional, Dict, List
import numpy as np
import json
from diffusion import Diffusion, compute_gaussian_product_coef, get_multidiffusion_vf, multidiffusion_pad_inputs, multidiffusion_unpad_outputs
from networks import SinusoidalTemporalEmbedding
from plotting_utils import plot_spec_to_numpy
import torchaudio
import inspect
from audio_utils import phase_channels_to_R, stft_mag_R_to_wav, phase_R_to_channels
from audio_transforms.transforms import apply_audio_transforms
from fast_inference_optimizer import FastInferenceOptimizer

# SSR Eval - opsiyonel
try:
    import ssr_eval
    SSR_EVAL_AVAILABLE = True
except ImportError:
    print("⚠️ ssr_eval modülü bulunamadı. Test metrikleri devre dışı.")
    SSR_EVAL_AVAILABLE = False

from collections import defaultdict, OrderedDict
import copy
import os
from scipy.io.wavfile import write as write_wav
from utils import find_middle_of_zero_segments

from tqdm import tqdm


class FastTimePartitionedPretrainedSTFTBridgeModel(LightningModule):
    """
    Hızlandırılmış inference için optimize edilmiş model
    
    Yeni özellikler:
    - torch.compile() desteği
    - Mixed precision (FP16/BF16)
    - Optimized CUDA settings
    - Batch processing optimizations
    - Memory efficient inference
    """
    
    def __init__(self, vf_model: torch.nn.Module,
                 inv_transforms=[], sampling_rate=22050,
                 n_timestep_channels=128,
                 beta_max=0.3, use_ot_ode=False,
                 fast_inpaint_mode=False,
                 pretrained_checkpoints: List[str]=None,
                 t_cutoffs: List[float]=[0.5],
                 predict_n_steps=50,
                 predict_hop_length=128,
                 predict_win_length=256,
                 predict_batch_size=8,
                 output_audio_filename="recon.wav",
                 # Fast inference parametreleri
                 use_fast_inference=True,
                 use_compile=True,
                 use_mixed_precision=True,
                 precision="fp16",  # "fp16", "bf16", "fp32"
                 compile_mode="reduce-overhead",
                 use_cudnn_benchmark=True,
                 use_tf32=True
                 ):
        super().__init__()
        self.predict_output_dir = os.path.dirname(output_audio_filename)
        for item in inspect.signature(FastTimePartitionedPretrainedSTFTBridgeModel).parameters:
            setattr(self, item, eval(item))
        self.ddpm = Diffusion(beta_max=beta_max)
        self.t_to_emb = SinusoidalTemporalEmbedding(n_bands=int(n_timestep_channels//2), min_freq=0.5)
        
        if self.use_ot_ode:
            print("Using ODE formulation")
        else:
            print("Using SDE formulation")
            
        self.test_results = defaultdict(list)
        assert(len(t_cutoffs) + 1 == len(pretrained_checkpoints))
        self.load_t_bounded_checkpoints(pretrained_checkpoints, t_cutoffs)
        self.fast_inpaint_mode = fast_inpaint_mode
        
        # Fast inference optimizer
        if use_fast_inference:
            self.optimizer = FastInferenceOptimizer(
                use_compile=use_compile,
                use_mixed_precision=use_mixed_precision,
                precision=precision,
                use_cudnn_benchmark=use_cudnn_benchmark,
                use_tf32=use_tf32,
                compile_mode=compile_mode,
                quantize=False  # Quantization diffusion modellerde dikkatli kullanılmalı
            )
            print("✓ Fast inference optimizer initialized")
        else:
            self.optimizer = None

    @torch.no_grad()
    def load_t_bounded_checkpoints(self, pretrained_checkpoints, t_cutoffs):
        loaded_models = []
        for ckpt in pretrained_checkpoints:
            state_dict = torch.load(ckpt, map_location='cpu')['state_dict']
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                if 'vf_model' in key:
                    new_key = key.replace('vf_model.', '')
                    new_state_dict[new_key] = value
            current_model = copy.deepcopy(self.vf_model)
            current_model.load_state_dict(new_state_dict)
            loaded_models.append(current_model)
        self.t_bounded_pretrained_models = nn.ModuleList(loaded_models)

    def optimize_models_for_inference(self):
        """Tüm modelleri inference için optimize et"""
        if self.optimizer is None:
            return
            
        print("\n" + "="*60)
        print("Optimizing models for fast inference...")
        print("="*60)
        
        optimized_models = []
        for idx, model in enumerate(self.t_bounded_pretrained_models):
            print(f"\nOptimizing model {idx+1}/{len(self.t_bounded_pretrained_models)}...")
            optimized_model = self.optimizer.optimize_model(model)
            optimized_models.append(optimized_model)
            
        self.t_bounded_pretrained_models = nn.ModuleList(optimized_models)
        print("\n✓ All models optimized!")
        print("="*60 + "\n")

    @torch.no_grad()
    def get_vf_model(self, t: float):
        model_idx = 0
        for idx, thresh in enumerate(self.t_cutoffs):
            if t >= thresh:
                model_idx = idx + 1
        return self.t_bounded_pretrained_models[model_idx]

    @torch.no_grad()
    def vocode_stft(self, spec_out):
        """
        spec_out: B x C x H x W model outputs to be mapped back to waveform
        """
        all_wav_samples = []
        num_samples = spec_out.shape[0]

        for b in range(num_samples):
            all_wav_samples.append(apply_audio_transforms(spec_out[b], self.inv_transforms)[0])

        return all_wav_samples

    @torch.inference_mode()  # Daha hızlı than @torch.no_grad()
    def ddpm_sample_fast(self, x_1, t_steps=None, mask=None, mask_pred_x0=True,
                    win_length=256,
                    hop_length=256,
                    batch_size=8
                    ):
        """
        Hızlandırılmış DDPM sampling
        
        Optimizasyonlar:
        - torch.inference_mode() kullanımı
        - Mixed precision support
        - Efficient memory management
        - Optimized batch processing
        """
        n_steps = t_steps.shape[1] - 1
        original_width = x_1.shape[-1]
        x_1 = multidiffusion_pad_inputs(x_1, win_length, hop_length)
        mask = multidiffusion_pad_inputs(mask, win_length, hop_length)

        x_t = x_1.clone()
        pred_x0 = None
        all_pred_x0s = []
        
        # Mixed precision context
        use_amp = self.optimizer is not None and self.optimizer.use_mixed_precision
        dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16 if self.precision == "bf16" else torch.float32

        for t_idx in tqdm(range(n_steps), desc="Sampling", disable=False):
            t_emb = self.t_to_emb(t_steps[:,t_idx]).repeat(x_1.shape[0], 1)
            t = t_steps[:, t_idx]
            t_prev = t_steps[:, t_idx+1]
            
            vf_model = self.get_vf_model(t[0].item())
            
            # Mixed precision inference
            if use_amp and torch.cuda.is_available():
                with torch.cuda.amp.autocast(dtype=dtype):
                    vf_output = get_multidiffusion_vf(
                        vf_model, x_t, t_emb, 
                        win_length=win_length,
                        hop_length=hop_length, 
                        batch_size=batch_size
                    )
            else:
                vf_output = get_multidiffusion_vf(
                    vf_model, x_t, t_emb, 
                    win_length=win_length,
                    hop_length=hop_length, 
                    batch_size=batch_size
                )
            
            pred_x0 = self.ddpm.get_pred_x0(t_steps[:, t_idx], x_t, vf_output)
            
            if mask is not None and mask_pred_x0:
                pred_x0 = pred_x0 * mask + (1-mask) * x_1

            all_pred_x0s.append(pred_x0.cpu())
            x_t_prev = self.ddpm.p_posterior(t_prev, t, x_t, pred_x0, ot_ode=self.use_ot_ode)
            x_t = x_t_prev
            
            if mask is not None:
                xt_true = x_1
                if not self.use_ot_ode:
                    std_sb = self.ddpm.get_std_t(t_prev)
                    xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                x_t = (1. - mask) * xt_true + mask * x_t
                
            # Memory management
            if t_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        all_pred_x0s = [multidiffusion_unpad_outputs(pred, original_width) for pred in all_pred_x0s]
        return all_pred_x0s
    
    @torch.inference_mode()
    def fast_inpaint_ddpm_sample_fast(self, x_1, t_steps=None, mask=None, mask_pred_x0=True,
                    win_length=256,
                    hop_length=256,
                    batch_size=8):
        """
        Hızlandırılmış fast inpaint sampling
        """
        original_width = x_1.shape[-1]
        x_1 = x_1.clone()
        x_1 = multidiffusion_pad_inputs(x_1, win_length, hop_length)
        mask = multidiffusion_pad_inputs(mask, win_length, hop_length, padding_constant=0)

        middle_indices = find_middle_of_zero_segments(1-mask[0,0,0])
        
        for center_idx in tqdm(middle_indices, desc="Fast inpainting"):
            l_idx = int(center_idx-win_length/2)
            r_idx = int(center_idx +win_length/2)
            if l_idx < 0:
                r_idx -= l_idx
                l_idx = 0

            if r_idx > x_1.shape[-1]:
                l_idx -= (r_idx - x_1.shape[-1])
                r_idx = x_1.shape[-1]
                
            assert(r_idx - l_idx == win_length)
            assert(l_idx >= 0)
            assert(r_idx <= x_1.shape[-1])
            
            curr_x_1 = x_1[:,:,:,l_idx:r_idx]
            curr_mask = mask[:,:,:,l_idx:r_idx]
            new_x_0 = self.ddpm_sample_fast(
                curr_x_1, t_steps, mask=curr_mask, 
                mask_pred_x0=mask_pred_x0, 
                win_length=win_length, 
                hop_length=hop_length, 
                batch_size=batch_size
            )
            x_1[:,:,:,l_idx:r_idx] = new_x_0[-1]
            
        x_1 = multidiffusion_unpad_outputs(x_1, original_width)
        return [x_1]
    
    @torch.inference_mode()
    def predict_step(self, batch, batch_idx):
        bs, channels, height, width = batch['x_0_clean'].shape
        assert(bs == 1)  # only supports batch size 1 for now

        current_out_dir = os.path.join(self.predict_output_dir, batch['outdir'][0])
        os.makedirs(current_out_dir, exist_ok=True)
        
        x_0_clean = batch['x_0_clean']
        x_0_corrupted = batch['x_0_corrupted']
        mask = batch['loss_mask']
        t_steps = torch.linspace(1, 0.05, int(self.predict_n_steps)).unsqueeze(0).to(x_0_corrupted.device)
        
        # Hızlı sampling kullan
        if not self.fast_inpaint_mode:
            x_0s = self.ddpm_sample_fast(
                x_0_corrupted, t_steps=t_steps, mask=mask,
                mask_pred_x0=True, 
                win_length=self.predict_win_length, 
                hop_length=self.predict_hop_length,
                batch_size=self.predict_batch_size
            )
        else:
            x_0s = self.fast_inpaint_ddpm_sample_fast(
                x_0_corrupted, t_steps=t_steps, mask=mask,
                mask_pred_x0=True, 
                win_length=self.predict_win_length, 
                hop_length=self.predict_hop_length,
                batch_size=self.predict_batch_size
            )

        reconstructed_audio = self.vocode_stft(x_0s[-1].cpu())[0].cpu().data.numpy()
        input_audio = self.vocode_stft(x_0_corrupted.cpu())[0].cpu().data.numpy()
        
        write_wav(self.output_audio_filename, batch['output_sr'], reconstructed_audio)
        
        print(f"\n✓ Audio saved to: {self.output_audio_filename}")


class LogValidationInpaintingSTFTCallback(Callback):
    def get_mag(self, spec):
        if spec.shape[-3] == 2:
            return torch.sqrt((spec ** 2).sum(-3))
        else:
            return spec[:,0]

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                batch_idx, dataloader_idx=0):
        
        val_dataset = trainer.val_dataloaders[0].dataset
        num_to_plot = 0
        if pl_module.global_rank == 0 and batch_idx == 0:
            for i in range(num_to_plot):
                sample = val_dataset[i]
                sample_id = "Validation sample " + str(i) + "/"
                seq_lens = sample['seq_lens']

                x_0_corrupted = sample['x_0_corrupted'].unsqueeze(0).to(pl_module.device)
                x_0_corrupted_mag = pl_module.inv_transforms[0](self.get_mag(x_0_corrupted))
                x_0_clean = sample['x_0_clean'].unsqueeze(0).to(pl_module.device)
                x_0_clean_mag = pl_module.inv_transforms[0](self.get_mag(x_0_clean))
                mask = sample['loss_mask'].unsqueeze(0).to(pl_module.device)
                gt_reconstruction = pl_module.vocode_stft(x_0_clean.unsqueeze(0).cpu())

                pl_module.logger.experiment.add_audio(sample_id + "Original Audio",
                                                      gt_reconstruction[0].data.numpy(), pl_module.global_step,
                                                      pl_module.sampling_rate)

                pl_module.logger.experiment.add_image(sample_id + "Original Magnitude",
                                                      plot_spec_to_numpy(x_0_clean_mag[0].data.cpu().numpy()),
                                                      pl_module.global_step, dataformats="HWC")
                pl_module.logger.experiment.add_image(sample_id + "Masked Magnitude",
                                                      plot_spec_to_numpy(x_0_corrupted_mag[0].data.cpu().numpy()),
                                                      pl_module.global_step, dataformats="HWC")

                n_steps = 25
                t_steps = torch.linspace(1,0.05, n_steps).unsqueeze(0).to(x_0_corrupted.device)
                
                # Hızlı sampling kullan
                if hasattr(pl_module, 'ddpm_sample_fast'):
                    x_0s = pl_module.ddpm_sample_fast(x_0_corrupted.unsqueeze(0), mask=mask, t_steps=t_steps)
                else:
                    x_0s = pl_module.ddpm_sample(x_0_corrupted.unsqueeze(0), mask=mask, t_steps=t_steps)
                    
                sampled_spec = x_0s[-1]
                sampled_spec_mag = self.get_mag(pl_module.inv_transforms[0](sampled_spec[0]))
                pl_module.logger.experiment.add_image(sample_id + "Inpainted Magnitude",
                                                      plot_spec_to_numpy(sampled_spec_mag.data.cpu().numpy()),
                                                      pl_module.global_step, dataformats="HWC")
                inpainted_audio = pl_module.vocode_stft(sampled_spec[0:1].cpu())
                pl_module.logger.experiment.add_audio(sample_id + "Inpainted Audio",
                                                      inpainted_audio[0].data.numpy(), pl_module.global_step,
                                                      pl_module.sampling_rate)
