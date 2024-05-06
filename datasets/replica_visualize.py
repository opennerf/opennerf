"""
Visualize the replica predictions and ground truth.
"""

from pathlib import Path
from typing import Optional, List
import numpy as np
import json
import pyviz3d.visualizer as viz
import open3d as o3d
import os
import replica
import subprocess


def process_txt(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines


def main(scene):

  prefix = '/home/fengelmann/'
  replica_prefix = 'datasets/replica_gt_semantics/'
  output_dir = f'{prefix}opennerf_vis/{scene}'
  os.makedirs(output_dir, exist_ok=True)

  if False:
    # Read mesh
    mesh_path = f'data/nerfstudio/replica_{scene}/{scene}_mesh.ply'
    scene_point_cloud = o3d.io.read_point_cloud(str(mesh_path))
    scene_mesh = o3d.io.read_triangle_mesh(str(mesh_path))

    # Read grountruth semantic labels
    semantic_gt_labels_path = Path(replica_prefix) / f'semantic_labels_{scene}.txt'
    semantic_gt_labels = np.array([replica.map_to_reduced[int(s)] for s in process_txt(str(semantic_gt_labels_path))])
    semantic_gt_colors = np.array([v for v in replica.SCANNET_COLOR_MAP_200.values()])[semantic_gt_labels]
  
    # Read predicted semantics
    v = viz.Visualizer()
    try:
      semantic_lerf_labels = np.load(f'{prefix}Programming/lerf_mine/replica/predictions/lerf/{scene}.npy')
      semantic_lerf_colors = np.array([v for v in replica.SCANNET_COLOR_MAP_200.values()])[semantic_lerf_labels]
      v.add_points('LERF', np.array(scene_point_cloud.points), np.array(semantic_lerf_colors), np.array(scene_point_cloud.normals), resolution=3)
    except:
      print('No LERF')

    semantic_openscene_labels = np.load(f'{prefix}Programming/lerf_mine/replica/predictions/openscene_2d3d_ensemble/{scene}.npy')
    semantic_opennerf_labels = np.load(f'{prefix}Programming/lerf_mine/replica/predictions/openreno/{scene}.npy')
    semantic_opennerf_labels[semantic_opennerf_labels == 256] = 1
    semantic_openscene_colors = np.array([v for v in replica.SCANNET_COLOR_MAP_200.values()])[semantic_openscene_labels]
    semantic_opennerf_colors = np.array([v for v in replica.SCANNET_COLOR_MAP_200.values()])[semantic_opennerf_labels]

    v.add_points('RGB', np.array(scene_point_cloud.points), np.array(scene_point_cloud.colors) * 255, np.array(scene_point_cloud.normals), resolution=3)
    v.add_points('GT', np.array(scene_point_cloud.points), np.array(semantic_gt_colors), np.array(scene_point_cloud.normals), resolution=3)
    v.add_points('OpenScene', np.array(scene_point_cloud.points), np.array(semantic_openscene_colors), np.array(scene_point_cloud.normals), resolution=3)
    v.add_points('OpenNeRF (ours)', np.array(scene_point_cloud.points), np.array(semantic_opennerf_colors), np.array(scene_point_cloud.normals), resolution=3)
    v.save(output_dir)

    # Save as ply meshes (faster to visualize than point clouds in blender)
    o3d.io.write_triangle_mesh(f'{output_dir}_rgb.ply', scene_mesh)
    scene_mesh.vertex_colors = o3d.utility.Vector3dVector(semantic_gt_colors / 255.0)
    o3d.io.write_triangle_mesh(f'{output_dir}_gt.ply', scene_mesh)
    scene_mesh.vertex_colors = o3d.utility.Vector3dVector(semantic_openscene_colors / 255.0)
    o3d.io.write_triangle_mesh(f'{output_dir}_openscene.ply', scene_mesh)
    scene_mesh.vertex_colors = o3d.utility.Vector3dVector(semantic_opennerf_colors / 255.0)
    o3d.io.write_triangle_mesh(f'{output_dir}_opennerf.ply', scene_mesh)
    scene_mesh.vertex_colors = o3d.utility.Vector3dVector(semantic_lerf_colors / 255.0)
    o3d.io.write_triangle_mesh(f'{output_dir}_lerf.ply', scene_mesh)

    # Render with blender
    for sub in ['opennerf', 'rgb', 'gt', 'lerf', 'openscene']:
      v = viz.Visualizer(position=[1, 1, 1], focal_length=18, animation=True)
      path = f'{prefix}opennerf_vis/{scene}_{sub}.ply'
      pc = o3d.io.read_point_cloud(path)
      size = (np.max(np.array(pc.points), axis=0) - np.min(np.array(pc.points), axis=0))
      min = np.min(np.array(pc.points), axis=0)
      v.add_mesh(f'office_c_{sub}', path, translation=-min-size/2.0)
      blender_args = {'output_prefix': f'{prefix}/op/{scene}/{sub}/{sub}_', 'executable_path': '/home/fengelmann/blender/blender'}
      v.save('opennerf_vis', blender_args=blender_args)
  
  # Process scene videos
  path = f'/home/fengelmann/op/'
  scene
  subs = ['gt', 'lerf', 'openscene', 'opennerf']
  captions = ['Groundtruth', 'LERF', 'OpenScene', 'OpenNeRF (Ours)']
  for i, sub in enumerate(subs):
    input = os.path.join(path, scene, sub, f'{sub}_')
    output = os.path.join(path, scene, sub, f'{sub}.mp4')
    subprocess.run(["ffmpeg", "-y", "-i", f'{input}%04d.png', "-vcodec", "libx264", "-vf", "format=yuv420p", "-y", output])
    output_text = os.path.join(path, scene, sub, f'{sub}_text.mp4')
    subprocess.run(
      ["ffmpeg", "-y", "-i", output, "-vf",
      f"drawtext=fontfile=/path/to/font.ttf:text='{captions[i]}':fontcolor=white:fontsize=24:box=1:boxcolor=black@0.5:boxborderw=15:x=(w-text_w-15)/2.0:y=(text_h+3)",
      "-codec:a", "copy", output_text])

  # Vertical stacking
  input00 = os.path.join(path, scene, 'gt', f'gt_text.mp4')
  input01 = os.path.join(path, scene, 'lerf', f'lerf_text.mp4')
  input10 = os.path.join(path, scene, 'openscene', f'openscene_text.mp4')
  input11 = os.path.join(path, scene, 'opennerf', f'opennerf_text.mp4')
  output0 = os.path.join(path, scene, f'{scene}_0.mp4')
  output1 = os.path.join(path, scene, f'{scene}_1.mp4')
  output2 = os.path.join(path, scene, f'{scene}.mp4')
  subprocess.run(["ffmpeg", "-y", "-i", input00, "-i", input01, "-filter_complex", "hstack=inputs=2", output0])
  subprocess.run(["ffmpeg", "-y", "-i", input10, "-i", input11, "-filter_complex", "hstack=inputs=2", output1])
  subprocess.run(["ffmpeg", "-y", "-i", output0, "-i", output1, "-filter_complex", "vstack=inputs=2", output2])


if __name__ == "__main__":
  for scene in replica.scenes:
    main(scene)
