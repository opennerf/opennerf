# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""
render.py
"""
from __future__ import annotations

import json
import os
import shutil
import cv2

import struct
import sys
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
import open3d as o3d
import pyviz3d.visualizer as viz

import mediapy as media
import numpy as np
import torch
import tyro
from jaxtyping import Float
from rich import box, style
from rich.panel import Panel
from rich.progress import (BarColumn, Progress, TaskProgressColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)
from rich.table import Table
from torch import Tensor
from typing_extensions import Annotated

from nerfstudio.cameras.camera_paths import get_interpolated_camera_path, get_spiral_path
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.model_components import renderers
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import colormaps, install_checks
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn
from nerfstudio.utils.scripts import run_command

import replica

def make_homo(points):
  ones = np.ones((points.shape[0], 1))
  points_homo = np.concatenate([points, ones], axis=1)
  return points_homo


def make_non_homo(points_homo):
  # Assuming N x 2, or N x 3, etc.
  last_dimension = points_homo.shape[1] - 1
  points = points_homo / points_homo[:, last_dimension : last_dimension + 1]
  return points[:, 0:last_dimension]


def invert_pose_matrix(pose_matrix):
  rotation_matrix = pose_matrix[:3, :3]
  translation_vector = pose_matrix[:3, 3]
  rotation_matrix_inverse = rotation_matrix.T
  translation_vector_inverse = -(rotation_matrix_inverse @ translation_vector)
  inverse_pose_matrix = np.eye(4)
  inverse_pose_matrix[:3, :3] = rotation_matrix_inverse
  inverse_pose_matrix[:3, 3] = translation_vector_inverse
  return inverse_pose_matrix


def pointcloud_from_rgbd(rgb, depth, pose, intrinics):
    pass


def project_points_to_image(points, depth, pose, intrinsics, distance_delta=10.0, band=0):
    """Project points to image."""
    pose_inv = invert_pose_matrix(pose)
    mirror_z = np.eye(4)
    mirror_z[2, 2] = -1
    mirror_y = np.eye(4)
    mirror_y[1, 1] = -1
    points_homo = make_homo(points)
    trafo_points = (mirror_y @ mirror_z @ pose_inv @ points_homo.T).T

    # v = viz.Visualizer()
    # v.add_points(f'points', points, points, visible=True)

    proj_points_homo = (intrinsics @ trafo_points.T[0:3, :]).T
    proj_points = np.round(make_non_homo(proj_points_homo)).astype(int)
    # Points outside image mask
    mask_x = np.logical_and(
        proj_points[:, 0] >= band, proj_points[:, 0] < intrinsics[0, 2] * 2 - band
    )
    mask_y = np.logical_and(
        proj_points[:, 1] >= band, proj_points[:, 1] < intrinsics[1, 2] * 2 - band
    )
    mask_z = trafo_points[:, 2] > 0  # negative is in front of the camera
    mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)

    # v.add_points(f'trafo_points', trafo_points[mask, 0:3], trafo_points[mask, 0:3]/100, visible=True)

    trafo_points_z = trafo_points[mask, 2]

    image_coords = proj_points[mask]  # N x 2
    image_coords_x = image_coords[:, 0]
    image_coords_y = image_coords[:, 1]
    proj_points_min_dist = depth[image_coords_y, image_coords_x]
    image_mask_dist = (
        np.abs(np.abs(np.squeeze(proj_points_min_dist)) - trafo_points_z) < distance_delta
    )
    # proj_point_colors = image[
    #     image_coords_y[image_mask_dist],
    #     image_coords_x[image_mask_dist],
    # ]
    mask[mask] = image_mask_dist
    
    # v.add_points(f'trafo_points', trafo_points[mask, 0:3], proj_point_colors.cpu().numpy() * 255, visible=True)
    # v.save(f'replica/proj/office0')

    return mask, image_coords_y[image_mask_dist], image_coords_x[image_mask_dist]


def get_path_from_json(camera_path: Dict[str, Any]) -> Cameras:
    """Takes a camera path dictionary and returns a trajectory as a Camera instance.

    Args:
        camera_path: A dictionary of the camera path information coming from the viewer.

    Returns:
        A Cameras instance with the camera path.
    """

    image_height = camera_path["h"]
    image_width = camera_path["w"]

    if "camera_type" not in camera_path:
        camera_type = CameraType.PERSPECTIVE
    elif camera_path["camera_type"] == "fisheye":
        camera_type = CameraType.FISHEYE
    elif camera_path["camera_type"] == "equirectangular":
        camera_type = CameraType.EQUIRECTANGULAR
    elif camera_path["camera_type"].lower() == "omnidirectional":
        camera_type = CameraType.OMNIDIRECTIONALSTEREO_L
    elif camera_path["camera_type"].lower() == "vr180":
        camera_type = CameraType.VR180_L
    else:
        camera_type = CameraType.PERSPECTIVE

    c2ws = []
    fxs = []
    fys = []
    for camera in camera_path["keyframes"]:  
        # pose
        matrix = [float(c) for c in camera["matrix"][1:-1].split(',')]
        c2w = torch.t(torch.tensor(matrix).view(4, 4))[:3]
        c2ws.append(c2w)
        if camera_type in [
            CameraType.EQUIRECTANGULAR,
            CameraType.OMNIDIRECTIONALSTEREO_L,
            CameraType.OMNIDIRECTIONALSTEREO_R,
            CameraType.VR180_L,
            CameraType.VR180_R,
        ]:
            fxs.append(image_width / 2)
            fys.append(image_height)
        else:
            # field of view
            fov = camera["fov"]
            focal_length = three_js_perspective_camera_focal_length(fov, image_height)
            fxs.append(focal_length)
            fys.append(focal_length)

    times = None

    camera_to_worlds = torch.stack(c2ws, dim=0)
    fx = torch.tensor(fxs)
    fy = torch.tensor(fys)
    return Cameras(
        fx=fx,
        fy=fy,
        cx=image_width / 2,
        cy=image_height / 2,
        camera_to_worlds=camera_to_worlds,
        camera_type=camera_type,
        times=times,
    )


def _render_trajectory_video(
    pipeline: Pipeline,
    cameras: Cameras,
    output_filename: Path,
    rendered_output_names: List[str],
    crop_data: Optional[CropData] = None,
    rendered_resolution_scaling_factor: float = 1.0,
    seconds: float = 5.0,
    output_format: Literal["images", "video", "npy"] = "video",
    image_format: Literal["jpeg", "png"] = "jpeg",
    jpeg_quality: int = 100,
    depth_near_plane: Optional[float] = None,
    depth_far_plane: Optional[float] = None,
    colormap_options: colormaps.ColormapOptions = colormaps.ColormapOptions(),
    project_to_pointcloud: bool = False,
) -> None:
    """Helper function to create a video of the spiral trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        output_filename: Name of the output file.
        rendered_output_names: List of outputs to visualise.
        crop_data: Crop data to apply to the rendered images.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        seconds: Length of output video.
        output_format: How to save output data.
        depth_near_plane: Closest depth to consider when using the colormap for depth. If None, use min value.
        depth_far_plane: Furthest depth to consider when using the colormap for depth. If None, use max value.
        colormap_options: Options for colormap.
    """
    CONSOLE.print("[bold green]Creating trajectory " + output_format)
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
    cameras = cameras.to(pipeline.device)

    fps = len(cameras) / seconds

    progress = Progress(
        TextColumn(":movie_camera: Rendering :movie_camera:"),
        BarColumn(),
        TaskProgressColumn(
            text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
            show_speed=True,
        ),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=False, compact=False),
        TimeElapsedColumn(),
    )
    output_image_dir = output_filename.parent / output_filename.stem
    if output_format == "images" or output_format == "npy":
        output_image_dir.mkdir(parents=True, exist_ok=True)
    if output_format == "video":
        # make the folder if it doesn't exist
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        # NOTE:
        # we could use ffmpeg_args "-movflags faststart" for progressive download,
        # which would force moov atom into known position before mdat,
        # but then we would have to move all of mdat to insert metadata atom
        # (unless we reserve enough space to overwrite with our uuid tag,
        # but we don't know how big the video file will be, so it's not certain!)

    with ExitStack() as stack:
        writer = None

        v = viz.Visualizer()
        # scene = str(output_image_dir).split('_')[-1]
        scene = output_image_dir.parent.parent.name.split('_')[-1]
        mesh_path = Path('data/nerfstudio/') / f'replica_{scene}' / f'{scene}_mesh.ply'
        scene_point_cloud = o3d.io.read_point_cloud(str(mesh_path))
        points = np.array(scene_point_cloud.points)
        colors = np.array(scene_point_cloud.colors)
        normals = np.array(scene_point_cloud.normals)

        colors_aggregate = np.zeros([points.shape[0], 3])
        semantics_aggregate = np.zeros([points.shape[0], len(replica.valid_class_ids)])  # aggregate predictions
        openseg_aggregate = np.zeros([points.shape[0], 768])
        count = np.zeros([points.shape[0]])  # number of predictions

        with progress:
            for camera_idx in progress.track(range(cameras.size), description=""):
                # if camera_idx > 10:  # debug
                #  break

                camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx, aabb_box=None)
                with torch.no_grad():
                    outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)

                rgb = outputs['rgb']  # N x 3
                openseg = outputs['openseg']  # N x 768

                transform = pipeline.datamanager.train_dataparser_outputs.as_dict()['dataparser_transform'].cpu().numpy()
                scale_factor = pipeline.datamanager.train_dataparser_outputs.as_dict()['dataparser_scale']                            
                transform_inv = invert_pose_matrix(transform)
                c2w = cameras.camera_to_worlds[camera_idx].cpu().numpy()
                c2w = np.concatenate([c2w, np.array([[0.0, 0.0, 0.0, 1.0]])])
                c2w[:3, 3] /= scale_factor
                c2w = transform_inv @ c2w
                intrinsics = pipeline.datamanager.train_dataset.cameras.get_intrinsics_matrices()[camera_idx].cpu().numpy()
                intrinsics *= rendered_resolution_scaling_factor
                intrinsics[2, 2] = 1.0
                depth = outputs["depth"].cpu().numpy() / scale_factor
                
                image_gt_path = f'data/nerfstudio/replica_{scene}/images/frame_{str(camera_idx + 1).zfill(5)}.jpg'
                depth_gt_path = f'data/nerfstudio/replica_{scene}/depths/frame_{str(camera_idx + 1).zfill(5)}.png'

                image_gt = media.read_image(image_gt_path)
                depth_gt = media.read_image(depth_gt_path, dtype=np.uint16) / (1000.0 * 6.58)  # why this value?
                # depth_gt = cv2.imread(depth_gt_path, cv2.IMREAD_UNCHANGED) / 1000.0
                if rendered_resolution_scaling_factor != 1:
                    new_height = int(image_gt.shape[0] * rendered_resolution_scaling_factor)
                    new_width = int(image_gt.shape[1] * rendered_resolution_scaling_factor)
                    image_gt = media.resize_image(image_gt, [new_height, new_width])
                    # depth_gt = media.resize_image(depth_gt, [new_height, new_width])
                    depth_gt = depth_gt[::int(1/rendered_resolution_scaling_factor), ::int(1/rendered_resolution_scaling_factor)]

                mask, image_coords_y, image_coords_x = project_points_to_image(points, depth_gt, c2w, intrinsics, distance_delta=0.1)
                colors_aggregate[mask] += rgb[image_coords_y, image_coords_x].cpu().numpy()
                openseg_aggregate[mask] += openseg[image_coords_y, image_coords_x].cpu().numpy()
                count[mask] += 1
                print(count[mask].shape)
                # v.add_points(f'colors_{camera_idx}', points[mask], rgb[image_coords_y, image_coords_x].cpu().numpy() * 255, normals[mask])

                continue

                # from matplotlib.pylab import plt
                # plt.imshow(depth_gt)
                # plt.savefig('depth_gt.png')
                # plt.imshow(depth)
                # plt.savefig('depth_pr.png')
                # dh = 2
                # dw = 3
                # d = np.reshape(np.array(range(dh * dw)), [dh, dw])
                dh, dw = depth.shape[0], depth.shape[1]

                def generate_pointcloud_method2(depth, cx, cy, fx, fy):
                    depth = np.squeeze(depth)
                    rows, cols = depth.shape
                    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
                    valid = True  # (depth > 0) & (depth < 255)
                    z = np.where(valid, depth, np.nan)
                    x = np.where(valid, z * (c - cx) / fx, 0)
                    y = np.where(valid, z * (r - cy) / fy, 0)
                    return np.reshape(np.dstack((x, y, z)), [-1, 3])

                dd = generate_pointcloud_method2(depth_gt, intrinsics[0, 2], intrinsics[1, 2], intrinsics[0, 0], intrinsics[1, 1])
                pose = c2w

                dd_homo = make_homo(dd)
                mirror_z = np.eye(4)
                mirror_z[2, 2] = -1
                mirror_y = np.eye(4)
                mirror_y[1, 1] = -1
                cam2world = cameras.camera_to_worlds[camera_idx].cpu().numpy()
                cam2world_dd = (cam2world @ mirror_y @ mirror_z @ dd_homo.T).T
                c2w_dd = (c2w @ mirror_y @ mirror_z @ dd_homo.T).T

                pose_inv = invert_pose_matrix(pose)
                points_homo = make_homo(points)
                trafo_points = (pose_inv @ points_homo.T).T

                v.add_points('rgb', points, colors * 255, normals)
                v.add_points(f'rgb_{camera_idx}', points[mask], proj_point_colors * 255, normals[mask], visible=True)
                v.add_points(f'cam2world_dd_{camera_idx}', cam2world_dd[:, 0:3], cam2world_dd[:, 0:3] * 255)
                v.add_points(f'c2w_dd_{camera_idx}', c2w_dd[:, 0:3], c2w_dd[:, 0:3] * 255)

                # v.add_points(f'rgbt_{camera_idx}', trafo_points[mask, 0:3], proj_point_colors * 255, normals[mask], visible=True)
                # v.add_points(f'dd_{camera_idx}', dd[:, 0:3], dd[:, 0:3] * 255)
                v.save(f'replica/proj/{scene}')
                print('asd')
                
                # v.add_points(f'dd_{camera_idx}', dd.T, depth_points[:, 0:3] * 255)
                # v.add_points(f'dd_{camera_idx}', (depth_points @ c2w)[:, 0:3], depth_points[:, 0:3] * 255)
                # semantics[mask] = proj_point_colors.cpu().numpy()
                # semantics_count[mask] += 1.0


        # render_image = []
        # for rendered_output_name in rendered_output_names:
        #     if rendered_output_name not in outputs and rendered_output_name != 'semantics':
        #         CONSOLE.rule("Error", style="red")
        #         CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
        #         CONSOLE.print(
        #             f"Please set --rendered_output_name to one of: {outputs.keys()}", justify="center"
        #         )
        #         sys.exit(1)
        #     if rendered_output_name != 'semantics':
        #       output_image = outputs[rendered_output_name]

        #     # is_depth = rendered_output_name.find("depth") != -1
        #     # if is_depth:
        #     #     output_image = (
        #     #         colormaps.apply_depth_colormap(
        #     #             output_image,
        #     #             accumulation=outputs["accumulation"],
        #     #             near_plane=depth_near_plane,
        #     #             far_plane=depth_far_plane,
        #     #             colormap_options=colormap_options,
        #     #         )
        #     #         .cpu()
        #     #         .numpy()
        #     #     )
        #     elif rendered_output_name == "semantics":
        #         relevancies = [k for k in outputs.keys() if k.startswith('relevancy')]
        #         relevancies = torch.cat([outputs[k] for k in relevancies], dim=-1)

        #         output_image = torch.argmax(relevancies, dim=-1, keepdim=True).float()
        #         output_image = output_image.cpu().numpy().astype(int)
        #         output_image = np.array([v for v in replica.SCANNET_COLOR_MAP_200.values()])[output_image]
        #         output_image = np.squeeze(output_image) / 255.0

        #         # output_image = (
        #         #     colormaps.apply_colormap(
        #         #         image=output_image,
        #         #         colormap_options=colormap_options,
        #         #     )
        #         #     .cpu()
        #         #     .numpy()x
        #         # )

        #         if project_to_pointcloud:
        #             transform = pipeline.datamanager.train_dataparser_outputs.as_dict()['dataparser_transform'].cpu().numpy()
        #             scale_factor = pipeline.datamanager.train_dataparser_outputs.as_dict()['dataparser_scale']                            
        #             transform_inv = invert_pose_matrix(transform)
        #             c2w = cameras.camera_to_worlds[camera_idx].cpu().numpy()
        #             c2w = np.concatenate([c2w, np.array([[0.0, 0.0, 0.0, 1.0]])])
        #             c2w[:3, 3] /= scale_factor
        #             c2w = transform_inv @ c2w
        #             intrinsics = pipeline.datamanager.train_dataset.cameras.get_intrinsics_matrices()[camera_idx].cpu().numpy()
        #             intrinsics *= rendered_resolution_scaling_factor
        #             intrinsics[2, 2] = 1.0
        #             depth = outputs["accumulation"].cpu().numpy() / scale_factor
        #             mask, proj_point_colors = project_points_to_image(points, relevancies, depth, c2w, intrinsics)

        #             semantics[mask] = proj_point_colors.cpu().numpy()
        #             semantics_count[mask] += 1.0

        #             # v.add_points(f'rgb_{camera_idx}', points[mask], proj_point_colors * 255, normals[mask])

        #             origin = c2w @ np.array([0, 0, 0, 1])
        #             v.add_arrow(f'{camera_idx};Arrow_1', start=origin, end=c2w @ np.array([0.1, 0.0, 0.0, 1]), color=np.array([255, 0, 0]), stroke_width=0.005, head_width=0.01)
        #             v.add_arrow(f'{camera_idx};Arrow_2', start=origin, end=c2w @ np.array([0.0, 0.1, 0.0, 1]), color=np.array([0, 255, 0]), stroke_width=0.005, head_width=0.01)
        #             v.add_arrow(f'{camera_idx};Arrow_3', start=origin, end=c2w @ np.array([0.0, 0.0, 0.1, 1]), color=np.array([0, 0, 255]), stroke_width=0.005, head_width=0.01)
        #     else:
        #         if output_format != "npy":
        #             output_image = (
        #                 colormaps.apply_colormap(
        #                     image=output_image,
        #                     colormap_options=colormap_options,
        #                 )
        #                 .cpu()
        #                 .numpy()
        #             )
        #         else:
        #             output_image = output_image.cpu().numpy()
        #             output_image = output_image / np.expand_dims((np.linalg.norm(output_image, axis=-1) + 1e-5), -1)
        #     render_image.append(output_image)
        # render_image = np.concatenate(render_image, axis=1)

        # if output_format == "npy":
        #     np.save(output_image_dir / f"{camera_idx:05d}.npy", np.float16(render_image))
        # if output_format == "images":
        #     if image_format == "png":
        #         media.write_image(output_image_dir / f"{camera_idx:05d}.png", render_image, fmt="png")
        #     if image_format == "jpeg":
        #         media.write_image(
        #             output_image_dir / f"{camera_idx:05d}.jpg", render_image, fmt="jpeg", quality=jpeg_quality
        #         )
        # if output_format == "video":
        #     if writer is None:
        #         render_width = int(render_image.shape[1])
        #         render_height = int(render_image.shape[0])
        #         writer = stack.enter_context(
        #             media.VideoWriter(
        #                 path=output_filename,
        #                 shape=(render_height, render_width),
        #                 fps=fps,
        #             )
        #         )
        #     writer.add_image(render_image)
        if True:
            count[count==0] = 1
            colors_aggregate /= np.reshape(count, [-1, 1])
            openseg_aggregate /= np.reshape(count, [-1, 1])

            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            principalComponents = pca.fit_transform(openseg_aggregate - np.mean(openseg_aggregate, axis=0, keepdims=True))
            openseg_colors = principalComponents - np.min(principalComponents, axis=0, keepdims=True)
            openseg_colors /= np.max(openseg_colors, axis=0, keepdims=True)

            with open(output_image_dir / 'openseg.npy', 'wb') as f:
                np.save(f, openseg_aggregate)

            v.add_points(f'colors', points, colors_aggregate * 255, normals, resolution=4)
            v.add_points(f'openseg', points, openseg_colors * 255, normals, resolution=4)
            blender_path = '/Applications/Blender.app/Contents/MacOS/Blender'
            v.save(output_image_dir / 'vis', show_in_blender=True, blender_executable_path=blender_path)

        if False:
            semantics_count[semantics_count==0] = 1
            semantics /= np.reshape(semantics_count, [-1, 1])
            semantic_classes = np.argmax(semantics, axis=-1)
            semantic_classes[np.max(semantics, axis=-1) < 0.0001] = 51
            semantic_colors = np.array([v for v in replica.SCANNET_COLOR_MAP_200.values()])[semantic_classes]
            v.add_points(f'semantics', points, semantic_colors, normals)
            v.save(f'replica/proj/{scene}')
            with open(f'replica/predictions/{scene}_relevancies.npy', 'wb') as f:
                np.save(f, semantics)
            with open(f'replica/predictions/{scene}.npy', 'wb') as f:
                np.save(f, semantic_classes)
            with open(f'replica/predictions/{scene}.txt', 'w') as f:
                np.savetxt(f, semantic_classes.astype(int), fmt='%d', delimiter='\n')

    table = Table(
        title=None,
        show_header=False,
        box=box.MINIMAL,
        title_style=style.Style(bold=True),
    )
    if output_format == "video":
        if cameras.camera_type[0] == CameraType.EQUIRECTANGULAR.value:
            CONSOLE.print("Adding spherical camera data")
            insert_spherical_metadata_into_file(output_filename)
        table.add_row("Video", str(output_filename))
    else:
        table.add_row("Images", str(output_image_dir))
    CONSOLE.print(Panel(table, title="[bold][green]:tada: Render Complete :tada:[/bold]", expand=False))


@dataclass
class CropData:
    """Data for cropping an image."""

    background_color: Float[Tensor, "3"] = torch.Tensor([0.0, 0.0, 0.0])
    """background color"""
    center: Float[Tensor, "3"] = torch.Tensor([0.0, 0.0, 0.0])
    """center of the crop"""
    scale: Float[Tensor, "3"] = torch.Tensor([2.0, 2.0, 2.0])
    """scale of the crop"""


def get_crop_from_json(camera_json: Dict[str, Any]) -> Optional[CropData]:
    """Load crop data from a camera path JSON

    args:
        camera_json: camera path data
    returns:
        Crop data
    """
    if "crop" not in camera_json or camera_json["crop"] is None:
        return None

    bg_color = camera_json["crop"]["crop_bg_color"]

    return CropData(
        background_color=torch.Tensor([bg_color["r"] / 255.0, bg_color["g"] / 255.0, bg_color["b"] / 255.0]),
        center=torch.Tensor(camera_json["crop"]["crop_center"]),
        scale=torch.Tensor(camera_json["crop"]["crop_scale"]),
    )


@dataclass
class BaseRender:
    """Base class for rendering."""

    load_config: Path
    """Path to config YAML file."""
    output_path: Path = Path("renders/output.mp4")
    """Path to output video file."""
    image_format: Literal["jpeg", "png"] = "jpeg"
    """Image format"""
    jpeg_quality: int = 100
    """JPEG quality"""
    downscale_factor: float = 1.0
    """Scaling factor to apply to the camera image resolution."""
    eval_num_rays_per_chunk: Optional[int] = None
    """Specifies number of rays per chunk during eval. If None, use the value in the config file."""
    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb"])
    """Name of the renderer outputs to use. rgb, depth, etc. concatenates them along y axis"""
    depth_near_plane: Optional[float] = None
    """Closest depth to consider when using the colormap for depth. If None, use min value."""
    depth_far_plane: Optional[float] = None
    """Furthest depth to consider when using the colormap for depth. If None, use max value."""
    colormap_options: colormaps.ColormapOptions = colormaps.ColormapOptions()
    """Colormap options."""


@dataclass
class RenderCameraPath(BaseRender):
    """Render a camera path generated by the viewer or blender add-on."""

    camera_path_filename: Path = Path("camera_path.json")
    """Filename of the camera path to render."""
    output_format: Literal["images", "video", "npy"] = "video"
    """How to save output data."""

    def main(self) -> None:
        """Main function."""
        _, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
        )

        # install_checks.check_ffmpeg_installed()

        with open(self.camera_path_filename, "r", encoding="utf-8") as f:
            camera_path = json.load(f)
        seconds = 1  # camera_path["seconds"]
        crop_data = get_crop_from_json(camera_path)
        camera_path = get_path_from_json(camera_path)

        if (
            camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value
            or camera_path.camera_type[0] == CameraType.VR180_L.value
        ):
            # temp folder for writing left and right view renders
            temp_folder_path = self.output_path.parent / (self.output_path.stem + "_temp")

            Path(temp_folder_path).mkdir(parents=True, exist_ok=True)
            left_eye_path = temp_folder_path / "render_left.mp4"

            self.output_path = left_eye_path

            if camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value:
                CONSOLE.print("[bold green]:goggles: Omni-directional Stereo VR :goggles:")
            else:
                CONSOLE.print("[bold green]:goggles: VR180 :goggles:")

            CONSOLE.print("Rendering left eye view")

        # add mp4 suffix to video output if none is specified
        if self.output_format == "video" and str(self.output_path.suffix) == "":
            self.output_path = self.output_path.with_suffix(".mp4")

        _render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            crop_data=crop_data,
            seconds=seconds,
            output_format=self.output_format,
            image_format=self.image_format,
            jpeg_quality=self.jpeg_quality,
            depth_near_plane=self.depth_near_plane,
            depth_far_plane=self.depth_far_plane,
            colormap_options=self.colormap_options,
        )

        if (
            camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value
            or camera_path.camera_type[0] == CameraType.VR180_L.value
        ):
            # declare paths for left and right renders

            left_eye_path = self.output_path
            right_eye_path = left_eye_path.parent / "render_right.mp4"

            self.output_path = right_eye_path

            if camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value:
                camera_path.camera_type[0] = CameraType.OMNIDIRECTIONALSTEREO_R.value
            else:
                camera_path.camera_type[0] = CameraType.VR180_R.value

            CONSOLE.print("Rendering right eye view")
            _render_trajectory_video(
                pipeline,
                camera_path,
                output_filename=self.output_path,
                rendered_output_names=self.rendered_output_names,
                rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
                crop_data=crop_data,
                seconds=seconds,
                output_format=self.output_format,
                image_format=self.image_format,
                jpeg_quality=self.jpeg_quality,
                depth_near_plane=self.depth_near_plane,
                depth_far_plane=self.depth_far_plane,
                colormap_options=self.colormap_options,
            )

            self.output_path = Path(str(left_eye_path.parent)[:-5] + ".mp4")

            if camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_R.value:
                # stack the left and right eye renders vertically for ODS final output
                ffmpeg_ods_command = ""
                if self.output_format == "video":
                    ffmpeg_ods_command = f'ffmpeg -y -i "{left_eye_path}" -i "{right_eye_path}" -filter_complex "[0:v]pad=iw:2*ih[int];[int][1:v]overlay=0:h" -c:v libx264 -crf 23 -preset veryfast "{self.output_path}"'
                    run_command(ffmpeg_ods_command, verbose=False)
                if self.output_format == "images":
                    # create a folder for the stacked renders
                    self.output_path = Path(str(left_eye_path.parent)[:-5])
                    self.output_path.mkdir(parents=True, exist_ok=True)
                    if self.image_format == "png":
                        ffmpeg_ods_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.png")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.png")}" -filter_complex vstack -start_number 0 "{str(self.output_path)+"//%05d.png"}"'
                    elif self.image_format == "jpeg":
                        ffmpeg_ods_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.jpg")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.jpg")}" -filter_complex vstack -start_number 0 "{str(self.output_path)+"//%05d.jpg"}"'
                    run_command(ffmpeg_ods_command, verbose=False)

                # remove the temp files directory
                if str(left_eye_path.parent)[-5:] == "_temp":
                    shutil.rmtree(left_eye_path.parent, ignore_errors=True)
                CONSOLE.print("[bold green]Final ODS Render Complete")
            else:
                # stack the left and right eye renders horizontally for VR180 final output
                self.output_path = Path(str(left_eye_path.parent)[:-5] + ".mp4")
                ffmpeg_vr180_command = ""
                if self.output_format == "video":
                    ffmpeg_vr180_command = f'ffmpeg -y -i "{right_eye_path}" -i "{left_eye_path}" -filter_complex "[1:v]hstack=inputs=2" -c:a copy "{self.output_path}"'
                    run_command(ffmpeg_vr180_command, verbose=False)
                if self.output_format == "images":
                    # create a folder for the stacked renders
                    self.output_path = Path(str(left_eye_path.parent)[:-5])
                    self.output_path.mkdir(parents=True, exist_ok=True)
                    if self.image_format == "png":
                        ffmpeg_vr180_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.png")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.png")}" -filter_complex hstack -start_number 0 "{str(self.output_path)+"//%05d.png"}"'
                    elif self.image_format == "jpeg":
                        ffmpeg_vr180_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.jpg")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.jpg")}" -filter_complex hstack -start_number 0 "{str(self.output_path)+"//%05d.jpg"}"'
                    run_command(ffmpeg_vr180_command, verbose=False)

                # remove the temp files directory
                if str(left_eye_path.parent)[-5:] == "_temp":
                    shutil.rmtree(left_eye_path.parent, ignore_errors=True)
                CONSOLE.print("[bold green]Final VR180 Render Complete")


@dataclass
class RenderInterpolated(BaseRender):
    """Render a trajectory that interpolates between training or eval dataset images."""

    pose_source: Literal["eval", "train"] = "eval"
    """Pose source to render."""
    interpolation_steps: int = 10
    """Number of interpolation steps between eval dataset cameras."""
    order_poses: bool = False
    """Whether to order camera poses by proximity."""
    frame_rate: int = 24
    """Frame rate of the output video."""
    output_format: Literal["images", "video", "npy"] = "video"
    """How to save output data."""

    def main(self) -> None:
        """Main function."""
        _, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
        )

        if self.pose_source == "eval":
            assert pipeline.datamanager.eval_dataset is not None
            cameras = pipeline.datamanager.eval_dataset.cameras
        else:
            assert pipeline.datamanager.train_dataset is not None
            cameras = pipeline.datamanager.train_dataset.cameras

        seconds = self.interpolation_steps * len(cameras) / self.frame_rate
        camera_path = get_interpolated_camera_path(
            cameras=cameras,
            steps=self.interpolation_steps,
            order_poses=self.order_poses,
        )

        _render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names[0].split(','),
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds=seconds,
            output_format=self.output_format,
            image_format=self.image_format,
            depth_near_plane=self.depth_near_plane,
            depth_far_plane=self.depth_far_plane,
            colormap_options=self.colormap_options,
        )


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[RenderCameraPath, tyro.conf.subcommand(name="camera-path")],
        Annotated[RenderInterpolated, tyro.conf.subcommand(name="interpolate")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)
