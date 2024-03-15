"""
OpenNerf DataManager
"""

from __future__ import annotations

import os.path as osp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import torch
import yaml
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.engine.callbacks import (TrainingCallback,
                                         TrainingCallbackAttributes)
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.misc import IterableWrapper
from rich.progress import Console

CONSOLE = Console(width=120)
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager, VanillaDataManagerConfig)

from opennerf.data.utils.dino_dataloader import DinoDataloader
from opennerf.data.utils.openseg_dataloader import OpenSegDataloader


@dataclass
class OpenNerfDataManagerConfig(VanillaDataManagerConfig):
    """OpenNerf DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: OpenNerfDataManager)
    clip_loss_weight: float = 0.1
    n_scales: int = 30
    max_scale: float = 1.5
    """maximum scale used to compute relevancy with"""
    num_lerf_samples: int = 24
    hashgrid_layers: Tuple[int] = (12, 12)
    hashgrid_resolutions: Tuple[Tuple[int]] = ((16, 128), (128, 512))
    hashgrid_sizes: Tuple[int] = (19, 19)

class OpenNerfDataManager(VanillaDataManager):
    config: OpenNerfDataManagerConfig

    def __init__(
        self,
        config: OpenNerfDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )

        images = [self.train_dataset[i]["image"].permute(2, 0, 1)[None, ...] for i in range(len(self.train_dataset))]
        images = torch.cat(images)

        cache_dir = f"outputs/{self.config.dataparser.data.name}"
        # clip_cache_path = Path(osp.join(cache_dir, f"clip_{self.image_encoder.name}"))
        dino_cache_path = Path(osp.join(cache_dir, "dino.npy"))
        openseg_cache_path = Path(osp.join(cache_dir, "openseg.npy"))

        print('openseg_cache_path', openseg_cache_path)

        self.dino_dataloader = DinoDataloader(
            image_list=images,
            device=self.device,
            cfg={"image_shape": list(images.shape[2:4])},
            cache_path=dino_cache_path,
        )

        image_pathes = self.train_dataset._dataparser_outputs.image_filenames
        self.openseg_dataloader = OpenSegDataloader(
            image_list=image_pathes,
            device=self.device,
            cfg={"image_shape": list(images.shape[2:4])},
            cache_path=openseg_cache_path,
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        batch["dino"] = self.dino_dataloader(ray_indices)
        batch["openseg"] = self.openseg_dataloader(ray_indices)
        # print('dino.shape', batch['dino'].shape)
        # print('openseg.shape', batch['openseg'].shape)
        return ray_bundle, batch
