from typing import Optional
import numpy as np
import open3d as o3d

def np2o3d(points: np.ndarray, normals: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        pc.normals = o3d.utility.Vector3dVector(normals)
    return pc
