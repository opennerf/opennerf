"""
OpenNerf config file.
"""

from __future__ import annotations

from opennerf.opennerf_datamanager import (
    OpenNerfDataManagerConfig,
)
from opennerf.opennerf_model import OpenNerfModelConfig
from opennerf.opennerf_pipeline import OpenNerfPipelineConfig

# from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification


opennerf_method = MethodSpecification(
    config=TrainerConfig(
        method_name="opennerf",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=OpenNerfPipelineConfig(
            datamanager=OpenNerfDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                # camera_optimizer=CameraOptimizerConfig(
                #    mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
                # ),
            ),
            model=OpenNerfModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                # LERF has a bunch more things here, some hyper parameter setups
                hashgrid_sizes=(19, 19),
                hashgrid_layers=(12, 12),
                hashgrid_resolutions=((16, 128), (128, 512)),
                # num_lerf_samples=24,
            ),
        ),
        optimizers={
            # TODO: consider changing optimizers depending on your custom method
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=30000),
            },
            "opennerf": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-9),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=4000),
            },
            # lerf has sth else here
            #
            # "camera_opt": {
                # "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                # "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            # },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="OpenNeRF.",
)
