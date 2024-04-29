from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type, Literal

import torch
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.viewer.server.viewer_elements import *
from torch.nn import Parameter

from opennerf.opennerf_field import OpenNerfField
from opennerf.opennerf_fieldheadnames import OpenNerfFieldHeadNames
from opennerf.opennerf_renderers import CLIPRenderer, MeanRenderer


@dataclass
class OpenNerfModelConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: OpenNerfModel)
    clip_loss_weight: float = 0.1
    dino_loss_weight: float = 0.0
    openseg_loss_weight: float = 1.0
    openseg_loss: Literal["Huber", "Cosine", "MSE"] = 'MSE'
    n_scales: int = 30
    max_scale: float = 1.5
    """maximum scale used to compute relevancy with"""
    num_opennerf_samples: int = 24
    hashgrid_layers: Tuple[int] = (12, 12)
    hashgrid_resolutions: Tuple[Tuple[int]] = ((16, 128), (128, 512))
    hashgrid_sizes: Tuple[int] = (19, 19)
    num_hidden_clip_layers: int = 1


class OpenNerfModel(NerfactoModel):
    config: OpenNerfModelConfig

    def populate_modules(self):
        super().populate_modules()

        self.renderer_clip = CLIPRenderer()
        self.renderer_mean = MeanRenderer()

        self.opennerf_field = OpenNerfField(
            self.config.hashgrid_layers,
            self.config.hashgrid_sizes,
            self.config.hashgrid_resolutions,
            self.config.num_hidden_clip_layers,
        )

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        ray_samples_list.append(ray_samples)

        nerfacto_field_outputs, outputs, weights = self._get_outputs_nerfacto(ray_samples)
        opennerf_weights, best_ids = torch.topk(weights, self.config.num_opennerf_samples, dim=-2, sorted=False)

        def gather_fn(tens):
            return torch.gather(tens, -2, best_ids.expand(*best_ids.shape[:-1], tens.shape[-1]))

        dataclass_fn = lambda dc: dc._apply_fn_to_fields(gather_fn, dataclass_fn)
        opennerf_samples: RaySamples = ray_samples._apply_fn_to_fields(gather_fn, dataclass_fn)

        weights_list.append(weights)
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        opennerf_field_outputs = self.opennerf_field.get_outputs(opennerf_samples)

        outputs["dino"] = self.renderer_mean(
            embeds=opennerf_field_outputs[OpenNerfFieldHeadNames.DINO], weights=opennerf_weights.detach()
        )
        outputs["openseg"] = self.renderer_mean(
            embeds=opennerf_field_outputs[OpenNerfFieldHeadNames.OPENSEG], weights=opennerf_weights.detach()
        )

        return outputs


    def _get_outputs_nerfacto(self, ray_samples: RaySamples):
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

        FieldHeadNames.UNCERTAINTY

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
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if self.training:
            unreduced_dino = self.config.dino_loss_weight * torch.nn.functional.mse_loss(
                outputs["dino"], batch["dino"], reduction="none")
            loss_dict["dino_loss"] = unreduced_dino.sum(dim=-1).nanmean()

            if self.config.openseg_loss == 'Huber':
                unreduced_openseg = self.config.openseg_loss_weight * torch.nn.functional.huber_loss(
                    outputs["openseg"], batch["openseg"], delta=1.25, reduction="none")
            elif self.config.openseg_loss == 'Cosine':
                unreduced_openseg = self.config.openseg_loss_weight * (1.0 - torch.nn.functional.cosine_similarity(
                    outputs["openseg"], batch["openseg"]))                
            elif self.config.openseg_loss == 'MSE':
                unreduced_openseg = self.config.openseg_loss_weight * torch.nn.functional.mse_loss(
                    outputs["openseg"], batch["openseg"], reduction="none")
            
            loss_dict["openseg_loss"] = unreduced_openseg.sum(dim=-1).nanmean()
        return loss_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["opennerf"] = list(self.opennerf_field.parameters())
        return param_groups
