"""
OpenNeRF Pipeline
"""

import typing
from dataclasses import dataclass, field
from typing import Literal, Optional, Type

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)

from opennerf.opennerf_datamanager import OpenNerfDataManager, OpenNerfDataManagerConfig
from opennerf.opennerf_model import OpenNerfModel, OpenNerfModelConfig
from opennerf.encoders.image_encoder import BaseImageEncoderConfig, BaseImageEncoder
# from nerfstudio.data.datamanagers.base_datamanager import (
#     DataManager,
#     DataManagerConfig,
# )



@dataclass
class OpenNerfPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: OpenNerfPipeline)
    """target class to instantiate"""
    datamanager: OpenNerfDataManagerConfig = OpenNerfDataManagerConfig()
    """specifies the datamanager config"""
    model: OpenNerfModelConfig = OpenNerfModelConfig()
    """specifies the model config"""
    # LERF also has "network" here, but we don't need that


class OpenNerfPipeline(VanillaPipeline):
    """OpenNerf Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """

    def __init__(
        self,
        config: OpenNerfPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode

        self.datamanager: OpenNerfDataManager = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            # LERF has some image encoder here
        )
        self.datamanager.to(device)

        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                OpenNerfModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            dist.barrier(device_ids=[local_rank])
