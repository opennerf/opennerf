"""
OpenNerf Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import open_clip
import torch
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.ray_samplers import PDFSampler
from nerfstudio.model_components.renderers import DepthRenderer
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.utils.colormaps import ColormapOptions, apply_colormap
from nerfstudio.viewer.server.viewer_elements import *
from torch.nn import Parameter

from opennerf.encoders.image_encoder import BaseImageEncoder
from opennerf.opennerf_field import OpenNerfField
from opennerf.opennerf_fieldheadnames import OpenNerfFieldHeadNames
from opennerf.opennerf_renderers import CLIPRenderer, MeanRenderer


@dataclass
class OpenNerfModelConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: OpenNerfModel)
    clip_loss_weight: float = 0.1
    n_scales: int = 30
    max_scale: float = 1.5
    """maximum scale used to compute relevancy with"""
    num_lerf_samples: int = 24
    hashgrid_layers: Tuple[int] = (12, 12)
    hashgrid_resolutions: Tuple[Tuple[int]] = ((16, 128), (128, 512))
    hashgrid_sizes: Tuple[int] = (19, 19)


class OpenNerfModel(NerfactoModel):
    """>>> OpenNerf Model."""

    config: OpenNerfModelConfig

    def populate_modules(self):
        super().populate_modules()

        self.renderer_clip = CLIPRenderer()
        self.renderer_mean = MeanRenderer()

        # self.image_encoder: BaseImageEncoder = self.kwargs["image_encoder"]
        # self.image_encoder: BaseImageEncoder = self.kwargs["image_encoder"]
        self.opennerf_field = OpenNerfField(
            self.config.hashgrid_layers,
            self.config.hashgrid_sizes,
            self.config.hashgrid_resolutions,
            clip_n_dims=512, #self.image_encoder.embedding_dim,
        )
        # self.opennerf_field = OpenNerfField(clip_n_dims=self.image_encoder.embedding_dim)

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        ray_samples_list.append(ray_samples)

        nerfacto_field_outputs, outputs, weights = self._get_outputs_nerfacto(ray_samples)
        lerf_weights, best_ids = torch.topk(weights, self.config.num_lerf_samples, dim=-2, sorted=False)

        def gather_fn(tens):
            return torch.gather(tens, -2, best_ids.expand(*best_ids.shape[:-1], tens.shape[-1]))

        dataclass_fn = lambda dc: dc._apply_fn_to_fields(gather_fn, dataclass_fn)
        lerf_samples: RaySamples = ray_samples._apply_fn_to_fields(gather_fn, dataclass_fn)

        # if self.training:
        #     with torch.no_grad():
        #         clip_scales = ray_bundle.metadata["clip_scales"]
        #         clip_scales = clip_scales[..., None]
        #         dist = (lerf_samples.frustums.get_positions() - ray_bundle.origins[:, None, :]).norm(
        #             dim=-1, keepdim=True
        #         )
        #     clip_scales = clip_scales * ray_bundle.metadata["height"] * (dist / ray_bundle.metadata["fy"])
        # else:
        #     clip_scales = torch.ones_like(lerf_samples.spacing_starts, device=self.device)

        # override_scales = (
        #     None if "override_scales" not in ray_bundle.metadata else ray_bundle.metadata["override_scales"]
        # )
        weights_list.append(weights)
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        opennerf_field_outputs = self.opennerf_field.get_outputs(lerf_samples)
        
        # if self.training:
            # outputs["clip"] = self.renderer_clip(
                # embeds=opennerf_field_outputs[OpenNerfFieldHeadNames.CLIP], weights=lerf_weights.detach()
            # )
        outputs["dino"] = self.renderer_mean(
            embeds=opennerf_field_outputs[OpenNerfFieldHeadNames.DINO], weights=lerf_weights.detach()
        )

        outputs["openseg"] = self.renderer_clip(
            embeds=opennerf_field_outputs[OpenNerfFieldHeadNames.OPENSEG], weights=lerf_weights.detach()
        )

        # if not self.training:
        #     with torch.no_grad():
        #         max_across, best_scales = self.get_max_across(
        #             lerf_samples,
        #             lerf_weights,
        #             lerf_field_outputs[LERFFieldHeadNames.HASHGRID],
        #             clip_scales.shape,
        #             preset_scales=override_scales,
        #         )
        #         outputs["raw_relevancy"] = max_across  # N x B x 1
        #         outputs["best_scales"] = best_scales.to(self.device)  # N
        return outputs

    def _get_outputs_nerfacto(self, ray_samples: RaySamples):
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        return field_outputs, outputs, weights


    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """
        Return dict with losses.
        - Weighted Huber loss on the CLIP features.
        - Weighted MSE loss on the DINO loss.
        """
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if self.training:
            # unreduced_clip = self.config.clip_loss_weight * torch.nn.functional.huber_loss(
            #     outputs["clip"], batch["clip"], delta=1.25, reduction="none"
            # )
            # loss_dict["clip_loss"] = unreduced_clip.sum(dim=-1).nanmean()
            unreduced_dino = torch.nn.functional.mse_loss(outputs["dino"], batch["dino"], reduction="none")
            loss_dict["dino_loss"] = unreduced_dino.sum(dim=-1).nanmean()
            
            unreduced_openseg = self.config.clip_loss_weight * torch.nn.functional.huber_loss(
                outputs["openseg"], batch["openseg"], delta=1.25, reduction="none"
            )
            loss_dict["openseg_loss"] = unreduced_openseg.sum(dim=-1).nanmean()
            # unreduced_openseg = torch.nn.functional.mse_loss(outputs["openseg"], batch["openseg"], reduction="none")
            # loss_dict["openseg_loss"] = unreduced_openseg.sum(dim=-1).nanmean()
        return loss_dict