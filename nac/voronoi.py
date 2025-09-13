import numpy as np
import open3d as o3d
from typing import List
from .visualization import show


def voronoi_from_points(point_cloud: o3d.geometry.PointCloud) -> List[o3d.geometry.PointCloud]:
    points = np.asarray(point_cloud.points).astype(np.float32)



    pcs = []



    return [point_cloud]
