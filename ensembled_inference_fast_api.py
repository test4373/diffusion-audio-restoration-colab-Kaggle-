# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for A2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
# FAST INFERENCE VERSION - Optimized for speed
# ---------------------------------------------------------------

from lightning.pytorch.cli import LightningCLI
from A2SB_lightning_module_fast import FastTimePartitionedPretrainedSTFTBridgeModel, LogValidationInpaintingSTFTCallback
from datasets.datamodule import STFTAudioDataModule
from lightning.pytorch.callbacks import ModelCheckpoint


class FastInpaintingAudioSBLightningCLI(LightningCLI):
    """
    Hızlandırılmış inference için Lightning CLI
    
    Kullanım:
        python ensembled_inference_fast_api.py predict \\
            -c configs/ensemble_2split_sampling.yaml \\
            -c configs/inference_files_upsampling.yaml \\
            --model.predict_n_steps=50 \\
            --model.use_fast_inference=True \\
            --model.use_compile=True \\
            --model.precision=fp16
    """
    
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint_callback")
        parser.add_lightning_class_args(LogValidationInpaintingSTFTCallback, "validation_inpainting_callback")
        parser.set_defaults({
            "checkpoint_callback.filename": "latest-epoch_{epoch}-iter_{global_step:.0f}",
            "checkpoint_callback.monitor": "global_step",
            "checkpoint_callback.mode": "max",
            "checkpoint_callback.every_n_train_steps": 1000,
            "checkpoint_callback.dirpath": "/debug",
            "checkpoint_callback.save_top_k": -1,
            "checkpoint_callback.auto_insert_metric_name": False
        })
        parser.link_arguments("checkpoint_callback.dirpath", "trainer.default_root_dir")


if __name__ == '__main__':
    cli = FastInpaintingAudioSBLightningCLI(
        FastTimePartitionedPretrainedSTFTBridgeModel, 
        STFTAudioDataModule, 
        save_config_kwargs={"overwrite": True}
    )
