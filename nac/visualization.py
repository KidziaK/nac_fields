import numpy as np
import open3d as o3d
from typing import Any
from torch import Tensor

def show(obj: Any) -> None:
    if isinstance(obj, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj)
        obj = pcd

    if isinstance(obj, Tensor):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj.detach().cpu().numpy())
        obj = pcd

    o3d.visualization.draw_geometries([obj], point_show_normal=True, mesh_show_back_face=True)
