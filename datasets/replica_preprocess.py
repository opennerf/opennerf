"""
Convert the replica dataset into nerfstudio format.
"""

from pathlib import Path
from typing import Optional, List
import numpy as np
import json
import open3d as o3d

import replica

from nerfstudio.process_data import process_data_utils, record3d_utils
from nerfstudio.process_data.process_data_utils import CAMERA_MODELS


def process_txt(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines


def process_replica(data: Path, output_dir: Path):
    """Process Replica data into a nerfstudio dataset.

    This script does the following:

    1. Scales images to a specified size.
    2. Converts Record3D poses into the nerfstudio format.
    """

    mesh_path = data / '..' / 'office0_mesh.ply'  # why do we need this?
    scene_point_cloud = o3d.io.read_point_cloud(str(mesh_path))
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

    summary_log = []

    replica_image_dir = data / "results"

    if not replica_image_dir.exists():
        raise ValueError(f"Image directory {replica_image_dir} doesn't exist")

    replica_image_filenames = []
    for f in replica_image_dir.iterdir():
        if f.stem.startswith('frame'):  # removes possible duplicate images (for example, 123(3).jpg)
            if f.suffix.lower() in [".jpg"]:
                replica_image_filenames.append(f)

    replica_image_filenames = sorted(replica_image_filenames)
    num_images = len(replica_image_filenames)
    idx = np.arange(num_images)
    if max_dataset_size != -1 and num_images > max_dataset_size:
        idx = np.round(np.linspace(0, num_images - 1, max_dataset_size)).astype(int)

    replica_image_filenames = list(np.array(replica_image_filenames)[idx])

    # Copy images to output directory
    copied_image_paths = process_data_utils.copy_images_list(
        replica_image_filenames,
        image_dir=image_dir,
        verbose=verbose,
        num_downscales=num_downscales,
    )
    num_frames = len(copied_image_paths)

    copied_image_paths = [Path("images/" + copied_image_path.name) for copied_image_path in copied_image_paths]
    summary_log.append(f"Used {num_frames} images out of {num_images} total")
    if max_dataset_size > 0:
        summary_log.append(
            "To change the size of the dataset add the argument [yellow]--max_dataset_size[/yellow] to "
            f"larger than the current value ({max_dataset_size}), or -1 to use all images."
        )

    traj_path = data / "traj.txt"
    replica_to_json(copied_image_paths, traj_path, output_dir, indices=idx, scene_point_cloud=scene_point_cloud)


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


if __name__ == "__main__":
    for scene in replica.scenes:
      data = f'data/Replica/{scene}'
      output_dir = f'data/nerfstudio/replica_{scene}'
      process_replica(Path(data), Path(output_dir))
