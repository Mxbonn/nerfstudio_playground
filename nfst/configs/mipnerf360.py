from pathlib import Path

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManagerConfig,
)
from nerfstudio.data.dataparsers.mipnerf360_dataparser import Mipnerf360DataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from nfst.models.mipnerf360 import MipNerf360ModelConfig

mipnerf360 = MethodSpecification(
    description="MipNerf360",
    config=TrainerConfig(
        method_name="mipnerf360",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=False,  # TODO changing to True results in an error
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=Mipnerf360DataParserConfig(
                    data=Path("/project_ghent/data/nerfstudio-data-mipnerf360/garden")
                ),
                train_num_rays_per_batch=1,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(  # TODO remove this?
                    mode="SO3xR3",
                    optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                    scheduler=ExponentialDecaySchedulerConfig(
                        lr_final=6e-6, max_steps=200000
                    ),
                ),
            ),
            model=MipNerf360ModelConfig(eval_num_rays_per_chunk=1 << 15),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="tensorboard",
    ),
)
