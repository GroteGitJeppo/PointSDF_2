import numpy as np
import open3d as o3d
import torch


def visualize_point_cloud(data_pos, window_name=""):
    points = data_pos.cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0, 0, 1])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


def visualize_point_clouds(data_pos1, data_pos2, window_name=""):
    points1 = data_pos1.cpu().numpy()
    points2 = data_pos2.cpu().numpy()

    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()

    pcd1.points = o3d.utility.Vector3dVector(points1)
    pcd2.points = o3d.utility.Vector3dVector(points2)

    pcd1.paint_uniform_color([0, 0, 1])
    pcd2.paint_uniform_color([1, 0, 0])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd1)
    vis.add_geometry(pcd2)
    vis.run()
    vis.destroy_window()
