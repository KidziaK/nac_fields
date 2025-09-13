import numpy as np
import open3d as o3d
from typing import Any

def show(obj: Any) -> None:
    if isinstance(obj, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj)
        obj = pcd

    o3d.visualization.draw_geometries([obj], point_show_normal=True, mesh_show_back_face=True)
