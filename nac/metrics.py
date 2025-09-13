import numpy as np
import open3d as o3d
from enum import Enum, auto

class ChamferDistanceMethod(Enum):
    L1 = auto()
    L2 = auto()


def chamfer_distance(
    mesh1: o3d.geometry.TriangleMesh,
    mesh2: o3d.geometry.TriangleMesh,
    n_points: int = 500000,
    method: ChamferDistanceMethod = ChamferDistanceMethod.L1
) -> float:
    pcd1 = mesh1.sample_points_uniformly(number_of_points=n_points)
    pcd2 = mesh2.sample_points_uniformly(number_of_points=n_points)

    dists_pcd1_to_pcd2 = pcd1.compute_point_cloud_distance(pcd2)
    dists_pcd2_to_pcd1 = pcd2.compute_point_cloud_distance(pcd1)

    dists_pcd1_to_pcd2 = np.asarray(dists_pcd1_to_pcd2)
    dists_pcd2_to_pcd1 = np.asarray(dists_pcd2_to_pcd1)

    match method:
        case ChamferDistanceMethod.L1:
            chamfer_dist = np.mean(dists_pcd1_to_pcd2 ** 2) + np.mean(dists_pcd2_to_pcd1 ** 2)
        case ChamferDistanceMethod.L2:
            chamfer_dist = np.mean(dists_pcd1_to_pcd2) + np.mean(dists_pcd2_to_pcd1)

    return chamfer_dist

def hausdorff_distance(mesh1: o3d.geometry.TriangleMesh, mesh2: o3d.geometry.TriangleMesh, n_points: int = 500000) -> float:
    pcd1 = mesh1.sample_points_uniformly(number_of_points=n_points)
    pcd2 = mesh2.sample_points_uniformly(number_of_points=n_points)

    dists_pcd1_to_pcd2 = pcd1.compute_point_cloud_distance(pcd2)
    dists_pcd2_to_pcd1 = pcd2.compute_point_cloud_distance(pcd1)

    dists_pcd1_to_pcd2 = np.asarray(dists_pcd1_to_pcd2)
    dists_pcd2_to_pcd1 = np.asarray(dists_pcd2_to_pcd1)

    hausdorff_dist = np.max([np.max(dists_pcd1_to_pcd2), np.max(dists_pcd2_to_pcd1)])

    return hausdorff_dist
