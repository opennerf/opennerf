"""
Convert the replica|iphone dataset into nerfstudio format.
"""

from pathlib import Path
from typing import Optional, List
import numpy as np
import json
import os
import pymeshlab
import shutil
import replica
# import scenefun3d
import open3d as o3d

from nerfstudio.process_data import process_data_utils
from nerfstudio.process_data.process_data_utils import CAMERA_MODELS
import matplotlib.pyplot as plt
import pyviz3d.visualizer as viz
from PIL import Image
import matplotlib.cm as cm
import tyro
import mediapy as media

def process_txt(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines


def process_replica_scene(data: Path, output_dir: Path, num_frames: int):
    """Process Replica data into a nerfstudio dataset.

    This script does the following:

    1. Scales images to a specified size.
    2. Converts Record3D poses into the nerfstudio format.
    """

    # convert mesh to triangle mesh (open3d can only read triangle meshes)
    mesh_path = data / '..' / f'{data.name}_mesh.ply'
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(mesh_path))
    ms.apply_filter('meshing_poly_to_tri')
    os.makedirs(output_dir, exist_ok=True)
    ms.save_current_mesh(str(output_dir / mesh_path.name), save_vertex_normal=True)

    verbose = True
    num_downscales = 3
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3
        will downscale the images by 2x, 4x, and 8x."""
    max_dataset_size = 200
    """Max number of images to train on. If the dataset has more, images will be sampled approximately evenly. If -1,
    use all images."""

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = output_dir / "depths"
    depth_dir.mkdir(parents=True, exist_ok=True)

    summary_log = []

    replica_image_dir = data / "results"

    if not replica_image_dir.exists():
        raise ValueError(f"Image directory {replica_image_dir} doesn't exist")

    replica_image_filenames = []
    replica_depth_filenames = []
    for f in replica_image_dir.iterdir():
        if f.stem.startswith('frame'):  # removes possible duplicate images (for example, 123(3).jpg)
            if f.suffix.lower() in [".jpg"]:
                replica_image_filenames.append(f)
        if f.stem.startswith('depth'):  # removes possible duplicate images (for example, 123(3).jpg)
            if f.suffix.lower() in [".png"]:
                replica_depth_filenames.append(f)

    replica_image_filenames = sorted(replica_image_filenames)
    replica_depth_filenames = sorted(replica_depth_filenames)
    assert(len(replica_image_filenames) == len(replica_depth_filenames))
    num_images = len(replica_image_filenames)

    idx = np.arange(num_images)
    if max_dataset_size != -1 and num_images > max_dataset_size:
        idx = np.round(np.linspace(0, num_images - 1, max_dataset_size)).astype(int)

    replica_image_filenames = list(np.array(replica_image_filenames)[idx])
    replica_depth_filenames = list(np.array(replica_depth_filenames)[idx])

    # Copy images to output directory
    copied_image_paths = process_data_utils.copy_images_list(
        replica_image_filenames,
        image_dir=image_dir,
        verbose=verbose,
        num_downscales=num_downscales,
    )
    copied_depth_paths = process_data_utils.copy_images_list(
        replica_depth_filenames,
        image_dir=depth_dir,
        verbose=verbose,
        num_downscales=num_downscales,
    )
    assert(len(copied_image_paths) == len(copied_depth_paths))
    num_frames = len(copied_image_paths)

    copied_image_paths = [Path("images/" + copied_image_path.name) for copied_image_path in copied_image_paths]
    summary_log.append(f"Used {num_frames} images out of {num_images} total")
    if max_dataset_size > 0:
        summary_log.append(
            "To change the size of the dataset add the argument [yellow]--max_dataset_size[/yellow] to "
            f"larger than the current value ({max_dataset_size}), or -1 to use all images."
        )

    traj_path = data / "traj.txt"
    replica_to_json(copied_image_paths, traj_path, output_dir, indices=idx)


def replica_to_json(images_paths: List[Path], trajectory_txt: Path, output_dir: Path, indices: np.ndarray) -> int:
    """Converts Replica's metadata and image paths to a JSON file.

    Args:
        images_paths: list if image paths.
        traj_path: Path to the Replica trajectory file.
        output_dir: Path to the output directory.
        indices: Indices to sample the metadata_path. Should be the same length as images_paths.

    Returns:
        The number of registered images.
    """

    assert len(images_paths) == len(indices)
    poses_data = process_txt(trajectory_txt)
    poses_data = np.array(
            [np.array(
                [float(v) for v in p.split()]).reshape((4, 4)) for p in poses_data]
        )

    # Set up rotation matrix
    rot_x = np.eye(4)
    a = np.pi
    rot_x[1, 1] = np.cos(a)
    rot_x[2, 2] = np.cos(a)
    rot_x[1, 2] = -np.sin(a)
    rot_x[2, 1] = np.sin(a)

    camera_to_worlds = poses_data[indices] @ rot_x

    frames = []
    for i, im_path in enumerate(images_paths):
        c2w = camera_to_worlds[i]
        frame = {
            "file_path": im_path.as_posix(),
            "transform_matrix": c2w.tolist(),
        }
        frames.append(frame)

    with open(trajectory_txt.parents[1] / 'cam_params.json') as file:
        cam_params = json.load(file)

    # Camera intrinsics
    focal_length = cam_params['camera']['fx']
    h = cam_params['camera']['h']
    w = cam_params['camera']['w']
    cx, cy = w / 2.0, h / 2.0

    out = {
        "fl_x": focal_length,
        "fl_y": focal_length,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "camera_model": CAMERA_MODELS["perspective"].name,
    }

    out["frames"] = frames
    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)
    return len(frames)


def process_iphone_scene(data: Path, output_dir: Path, num_frames: int):
    """Process iPhone data into a nerfstudio dataset.

    This script does the following:

    1. Scales images to a specified size.
    2. Converts iphone poses into the nerfstudio format.
    """

    verbose = True
    num_downscales = 3
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3
        will downscale the images by 2x, 4x, and 8x."""
    max_dataset_size = num_frames
    """Max number of images to train on. If the dataset has more, images will be sampled approximately evenly. If -1,
    use all images."""

    rot_x_90 = np.eye(3)
    a = np.pi / 2.0
    rot_x_90[1, 1] = np.cos(a)
    rot_x_90[2, 2] = np.cos(a)
    rot_x_90[1, 2] = -np.sin(a)
    rot_x_90[2, 1] = np.sin(a)
    T_iphone2nerfstudio = np.eye(4)
    T_iphone2nerfstudio[0:3, 0:3] = rot_x_90
    
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = output_dir / "depths"
    depth_dir.mkdir(parents=True, exist_ok=True)

    summary_log = []

    if not image_dir.exists():
        raise ValueError(f"Image directory {image_dir} doesn't exist")

    image_filenames = []
    depth_filenames = []
    for f in data.iterdir():
        if f.stem.startswith('frame'):  # removes possible duplicate images (for example, 123(3).jpg)
            if f.suffix.lower() in [".jpg"]:
                image_filenames.append(f)

    image_filenames = sorted(image_filenames)
    num_images = len(image_filenames)
    idx = np.arange(num_images)
    if max_dataset_size != -1 and num_images > max_dataset_size:
        idx = np.round(np.linspace(0, num_images - 1, max_dataset_size)).astype(int)

    image_filenames = list(np.array(image_filenames)[idx])

    # Iterate over images, render depth and save the depths
    mesh_path = data / 'textured_output.obj'
    texture_path = data / 'textured_output.jpg'
    texture = np.array(Image.open(str(texture_path)))
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))

    # Assign vertex colors to the mesh
    def texture_to_vertex_colors(mesh, texture):
        # Get UV coordinates and vertex positions
        uvs = np.asarray(mesh.triangle_uvs)
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        # Create an array to store vertex colors
        vertex_colors = np.zeros((len(vertices), 3))
        # Iterate over each triangle
        for tri_idx, tri in enumerate(triangles):
            uv_coords = uvs[3*tri_idx : 3*tri_idx + 3]
            vert_indices = tri
            # Get texture color for each vertex
            for i, uv in enumerate(uv_coords):
                # Ensure UV coordinates are within [0, 1]
                u, v = uv[0] % 1, uv[1] % 1
                # Convert UV coordinates to image coordinates
                img_x = int(np.round(u * (texture.shape[1] - 1)))
                img_y = int(np.round((1 - v) * (texture.shape[0] - 1)))
                # Get the color from the texture
                color = texture[img_y, img_x] / 255.0  # Normalize color to [0, 1]
                vertex_colors[vert_indices[i]] = color
        return vertex_colors

    # Save the mesh as a colored and meshed PLY file
    print('vertex colors...')
    vertex_colors = texture_to_vertex_colors(mesh, texture)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    mesh.compute_vertex_normals()

    v = viz.Visualizer(up=np.array([0.0, 0.0, 1.0]))

    # rotate mesh (from iPhone world to NerfStudio world) --> z-up, x-right, y-lookAt
    mesh_vertices_rotated = T_iphone2nerfstudio[0:3, 0:3] @ np.array(mesh.vertices).T
    mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices_rotated.T)

    point_cloud_path = output_dir / f'{data.name}_mesh.ply'
    o3d.io.write_triangle_mesh(str(point_cloud_path), mesh)
    v.add_points('scene', positions=np.array(mesh.vertices), colors=np.array(mesh.vertex_colors) * 255)

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultUnlit'

    # v.add_mesh('scene_full', str(point_cloud_path))

    img_width, img_height = Image.open(image_filenames[0]).size
    renderer_pc = o3d.visualization.rendering.OffscreenRenderer(img_width, img_height)
    renderer_pc.scene.set_background(np.array([0, 0, 0, 0]))
    renderer_pc.scene.add_geometry("mesh", mesh, mat)
    camera_parameters = o3d.camera.PinholeCameraParameters()

    focal_lengths = []
    img_widths = []
    img_heights = []
    frames = []
    for i, image_filename in enumerate(image_filenames):  # [::1][:1055]  #
        print()
        print(i, image_filename)
        with open(str(image_filename)[:-3]+'json') as file:
            cam_params = json.load(file)

        image_rgb = Image.open(image_filename)
        img_width, img_height = image_rgb.size

        rot_x_180 = np.eye(4)
        a = np.pi
        rot_x_180[1, 1] = np.cos(a)
        rot_x_180[2, 2] = np.cos(a)
        rot_x_180[1, 2] = -np.sin(a)
        rot_x_180[2, 1] = np.sin(a)

        extrinsics = T_iphone2nerfstudio @ np.array(cam_params['cameraPoseARFrame']).reshape(4, 4)

        # collect intrinsics, needed for average computation later
        img_widths.append(img_width)
        img_heights.append(img_height)
        focal_lengths.append(cam_params['intrinsics'][0])
        focal_lengths.append(cam_params['intrinsics'][4])

        rot_x_90 = np.eye(4)
        a = np.pi / 2.0
        rot_x_90[1, 1] = np.cos(a)
        rot_x_90[2, 2] = np.cos(a)
        rot_x_90[1, 2] = -np.sin(a)
        rot_x_90[2, 1] = np.sin(a)

        frame = {
            "file_path": f'images/frame_{str(i + 1).zfill(5)}.jpg',
            "transform_matrix": extrinsics.tolist(),
        }
        frames.append(frame)

        # Set camera intrinsic parameters
        intrinsics = o3d.camera.PinholeCameraIntrinsic()
        intrinsics.set_intrinsics(img_width, img_height,
                                  cam_params['intrinsics'][0], cam_params['intrinsics'][4],
                                  cam_params['intrinsics'][2], cam_params['intrinsics'][5])
        
        r = extrinsics[0:3, 0:3]
        t = extrinsics[0:3, 3]
        extrinsics_inv = np.eye(4)
        extrinsics_inv[0:3, 0:3] = r.T
        extrinsics_inv[0:3, 3] = -r.T @ t
        extrinsics_open3d = rot_x_180 @ extrinsics_inv

        camera_parameters.intrinsic = intrinsics
        camera_parameters.extrinsic = extrinsics_open3d
        renderer_pc.setup_camera(camera_parameters.intrinsic, camera_parameters.extrinsic)
        depth_image = np.asarray(renderer_pc.render_to_depth_image(z_in_view_space=True))

        depth_image[np.isinf(depth_image)] = 0.0
        depth_image *= 1000

        image = Image.fromarray(depth_image.astype(np.uint16))
        depth_filename = image_filename.with_suffix('.png')
        image.save(depth_filename)

        depth_filenames.append(depth_filename)

        # Visualization
        if False:
            v.add_arrow(f'{i};Arrow_x', start=extrinsics[0:3, :] @ np.array([0.0, 0.0, 0.0, 1.0]), end=extrinsics[0:3, :] @ np.array([0.1, 0.0, 0.0, 1.0]), color=np.array([255, 0, 0]), stroke_width=0.005, head_width=0.01)
            v.add_arrow(f'{i};Arrow_y', start=extrinsics[0:3, :] @ np.array([0.0, 0.0, 0.0, 1.0]), end=extrinsics[0:3, :] @ np.array([0.0, 0.1, 0.0, 1.0]), color=np.array([0, 255, 0]), stroke_width=0.005, head_width=0.01)
            v.add_arrow(f'{i};Arrow_z', start=extrinsics[0:3, :] @ np.array([0.0, 0.0, 0.0, 1.0]), end=extrinsics[0:3, :] @ np.array([0.0, 0.0, 0.1, 1.0]), color=np.array([0, 0, 255]), stroke_width=0.005, head_width=0.01)

            cmap = cm.get_cmap('viridis')  # You can choose other colormaps like 'plasma', 'inferno', etc.
            # normalized_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
            # rgb_map = cmap(depth_image)[:, :, 0:3]

            # For debugging only, visualize the saved depth map point cloud
            if True:
                depth_gt = media.read_image(depth_filename, dtype=np.uint16) / 1000.0
                height, width = depth_gt.shape
                ii, jj = np.indices((height, width))
                zz = depth_gt
                xx = (jj - cam_params['intrinsics'][2]) * zz / cam_params['intrinsics'][0]
                yy = (ii - cam_params['intrinsics'][5]) * zz / cam_params['intrinsics'][4]

                colors = np.array(image_rgb).reshape(-1, 3)[::500]
                points = np.stack((xx, yy, zz, np.ones(xx.shape)), axis=-1).reshape(-1, 4)[::500]
                points_trans = (np.linalg.inv(extrinsics_open3d)[0:3, :] @ points.copy().T).T
                # v.add_points(f'scene_{i}_orig', positions=points[:, 0:3])
                v.add_points(f'scene_{i}_t', positions=points_trans[:, 0:3], colors=colors)
        
            v.save('example_arrows')

    # Copy images to output directory
    copied_image_paths = process_data_utils.copy_images_list(
        image_filenames,
        image_dir=image_dir,
        verbose=verbose,
        num_downscales=num_downscales,
    )
    copied_depth_paths = process_data_utils.copy_images_list(
        depth_filenames,
        image_dir=depth_dir,
        verbose=verbose,
        num_downscales=num_downscales,
    )

    # assert(len(copied_image_paths) == len(copied_depth_paths))
    num_frames = len(copied_image_paths)

    copied_image_paths = [Path("images/" + copied_image_path.name) for copied_image_path in copied_image_paths]
    summary_log.append(f"Used {num_frames} images out of {num_images} total")
    if max_dataset_size > 0:
        summary_log.append(
            "To change the size of the dataset add the argument [yellow]--max_dataset_size[/yellow] to "
            f"larger than the current value ({max_dataset_size}), or -1 to use all images."
        )

    # save camera trajectroy json
    out = {
        "fl_x": np.mean(focal_lengths),
        "fl_y": np.mean(focal_lengths),
        "cx": np.mean(img_widths) / 2.0,
        "cy": np.mean(img_heights) / 2.0,
        "w": np.mean(img_widths),
        "h": np.mean(img_heights),
        "camera_model": CAMERA_MODELS["perspective"].name,
        "frames": frames,
    }
    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)


def process_scenefun3d_scene(data: Path, output_dir: Path, num_frames: int):
    """Process SceneFun3D data into a nerfstudio dataset.

    This script does the following:

    1. Scales images to a specified size.
    2. Converts SceneFun3D poses into the nerfstudio format.
    """


def main(dataset_name: str, num_frames: int = 200) -> None:

    scene_names = []
    if dataset_name == 'replica':
        scene_names = replica.scenes
        process_scene = process_replica_scene
    elif dataset_name == 'scenefun3d':
        # scene_names = scenefun3d.scenes
        process_scene = process_scenefun3d_scene
    elif dataset_name == 'iphone':
        scene_names = ['desk', 'people', 'spot']  # example scenes
        process_scene = process_iphone_scene
    
    for scene_name in scene_names:
      data = f'data/{dataset_name}/{scene_name}'
      output_dir = f'data/nerfstudio/{dataset_name}/{scene_name}'
      process_scene(Path(data), Path(output_dir), num_frames)

if __name__ == "__main__":
    tyro.cli(main)
