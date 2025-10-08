import torch
import open3d as o3d
import numpy as np
from torch import Tensor
from nac.visualization import show
from nac.settings import ReconstructionConfig
from scipy.spatial import cKDTree
from dataclasses import dataclass

@dataclass
class TrainingData:
    on_manifold_points: Tensor
    off_manifold_points: Tensor
    near_manifold_points: Tensor

class SirenDataset:
    def __init__(self, config: ReconstructionConfig, point_cloud: o3d.geometry.PointCloud):
        self.config = config
        points = np.asarray(point_cloud.points).astype(np.float32)
        normals = np.asarray(point_cloud.normals).astype(np.float32)

        kd_tree = cKDTree(points)
        dist, _ = kd_tree.query(points, k=51, workers=-1)
        sigmas = dist[:, -1:].astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas).to(config.device)
        self.points = torch.from_numpy(points).to(config.device)
        self.normals = torch.from_numpy(normals).to(config.device)

    def sample(self) -> TrainingData:
        n = self.config.samples
        device = self.config.device

        random_indices = torch.randperm(len(self.points))[:n]
        on_manifold_points = self.points[random_indices].to(device)

        off_manifold_points = 2 * (torch.rand(size=(n, 3)) -0.5).to(device)
        near_manifold_points = torch.normal(mean=on_manifold_points, std=self.sigmas[random_indices]).to(device)

        on_manifold_points.requires_grad = True
        off_manifold_points.requires_grad = True
        near_manifold_points.requires_grad = True

        return TrainingData(
            on_manifold_points=on_manifold_points,
            off_manifold_points=off_manifold_points,
            near_manifold_points=near_manifold_points
        )
