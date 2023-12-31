from pathlib import Path

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManagerConfig,
)
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from nfst.dataparsers.mipnerf360_dataparser import Mipnerf360DataParserConfig
from nfst.models.mipnerf360 import MipNerf360ModelConfig

mipnerf_steps=1000000
mipnerf360 = MethodSpecification(
    description="MipNerf360",
    config=TrainerConfig(
        method_name="mipnerf360",
        steps_per_eval_batch=500,
        steps_per_eval_image=0,
        steps_per_eval_all_images=50000,
        steps_per_save=10000,
        max_num_iterations=mipnerf_steps,
        mixed_precision=False,  # TODO changing to True results in an error
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=Mipnerf360DataParserConfig(
                    data=Path("/project_ghent/data/nerfstudio-data-mipnerf360/garden")
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=MipNerf360ModelConfig(eval_num_rays_per_chunk=4096),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=0.0005, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_pre_warmup=0.000005,
                    lr_final=0.000005,
                    warmup_steps=2048,
                    max_steps=mipnerf_steps,
                    ramp="linear",
                ),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=0.0005, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_pre_warmup=0.000005,
                    lr_final=0.000005,
                    warmup_steps=2048,
                    max_steps=mipnerf_steps,
                    ramp="linear",
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=4096),
        vis="tensorboard",
    ),
)
