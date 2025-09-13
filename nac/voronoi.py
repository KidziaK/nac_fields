import numpy as np
import open3d as o3d
from typing import List
from .utils import np2o3d
from .visualization import show
from .clustering import find_planar_clusters


def voronoi_from_points(point_cloud: o3d.geometry.PointCloud) -> List[o3d.geometry.PointCloud]:
    points = np.asarray(point_cloud.points).astype(np.float32)
    normals = np.asarray(point_cloud.normals).astype(np.float32)

    clusters_points, clusters_normals, unclassified_points, unclassified_normals = find_planar_clusters(
        points,
        normals,
        distance_threshold=0.001,
        max_iterations=2000,
        min_points_in_cluster=250
    )

    planar_pcs = [np2o3d(p, n) for p, n in zip(clusters_points, clusters_normals)]
    other_pcs = [np2o3d(unclassified_points, unclassified_normals)]

    return planar_pcs + other_pcs
