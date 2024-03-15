from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from jaxtyping import Float
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import (SceneContraction,
                                                             SpatialDistortion)
from nerfstudio.fields.base_field import Field
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from opennerf.opennerf_fieldheadnames import OpenNerfFieldHeadNames

try:
    import tinycudann as tcnn
except ImportError:
    pass


class OpenNerfField(Field):
    def __init__(
        self,
        grid_layers,
        grid_sizes,
        grid_resolutions,
        clip_n_dims: int,
        spatial_distortion: SpatialDistortion = SceneContraction(),
    ) -> None:

        super().__init__()
        assert len(grid_layers) == len(grid_sizes) and len(grid_resolutions) == len(grid_layers)
        self.spatial_distortion = spatial_distortion
        self.clip_encs = torch.nn.ModuleList(
            [
                OpenNerfField._get_encoding(
                    grid_resolutions[i][0], grid_resolutions[i][1], grid_layers[i], indim=3, hash_size=grid_sizes[i]
                )
                for i in range(len(grid_layers))
            ]
        )
        tot_out_dims = sum([e.n_output_dims for e in self.clip_encs])

        # self.clip_net = tcnn.Network(
        #     n_input_dims=tot_out_dims + 1,
        #     n_output_dims=clip_n_dims,
        #     network_config={
        #         "otype": "CutlassMLP",
        #         "activation": "ReLU",
        #         "output_activation": "None",
        #         "n_neurons": 256,
        #         "n_hidden_layers": 4,
        #     },
        # )

        self.dino_net = tcnn.Network(
            n_input_dims=tot_out_dims,
            n_output_dims=384,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 256,
                "n_hidden_layers": 1,
            },
        )

        self.openseg_net = tcnn.Network(
            n_input_dims=tot_out_dims,
            n_output_dims=768,  # this is the dimension of the openseg features
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 256,
                "n_hidden_layers": 1,
            },
        )


    @staticmethod
    def _get_encoding(start_res, end_res, levels, indim=3, hash_size=19):
        growth = np.exp((np.log(end_res) - np.log(start_res)) / (levels - 1))
        enc = tcnn.Encoding(
            n_input_dims=indim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": levels,
                "n_features_per_level": 8,
                "log2_hashmap_size": hash_size,
                "base_resolution": start_res,
                "per_level_scale": growth,
            },
        )
        return enc

    def get_outputs(self, ray_samples: RaySamples) -> Dict[OpenNerfFieldHeadNames, Float[Tensor, "bs dim"]]:
        # random scales, one scale
        outputs = {}

        positions = ray_samples.frustums.get_positions().detach()
        positions = self.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0

        xs = [e(positions.view(-1, 3)) for e in self.clip_encs]
        x = torch.concat(xs, dim=-1)

        outputs[OpenNerfFieldHeadNames.HASHGRID] = x.view(*ray_samples.frustums.shape, -1)

        # clip_pass = self.clip_net(torch.cat([x, clip_scales.view(-1, 1)], dim=-1)).view(*ray_samples.frustums.shape, -1)
        # outputs[OpenNerfFieldHeadNames.CLIP] = clip_pass / clip_pass.norm(dim=-1, keepdim=True)

        dino_pass = self.dino_net(x).view(*ray_samples.frustums.shape, -1)
        outputs[OpenNerfFieldHeadNames.DINO] = dino_pass

        openseg_pass = self.openseg_net(x).view(*ray_samples.frustums.shape, -1)
        outputs[OpenNerfFieldHeadNames.OPENSEG] = openseg_pass / openseg_pass.norm(dim=-1, keepdim=True)

        return outputs