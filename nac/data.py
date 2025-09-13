import torch
import open3d as o3d
import numpy as np
from dataclasses import dataclass
from scipy.spatial import cKDTree


@dataclass
class TrainingConfig:
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 10000
    learning_rate: float = 5e-5
    padding: float = 0.05
    surface_points_count: int = 10000
    off_surface_points_count: int = 10000
    near_surface_points_count: int = 10000


@dataclass
class TrainingData:
    surface_points: torch.Tensor
    surface_normals: torch.Tensor
    off_surface_points: torch.Tensor
    near_surface_points: torch.Tensor


class DataSampler:
    def __init__(self, point_cloud: o3d.geometry.PointCloud, config: TrainingConfig):
        self.config = config
        self.points = np.asarray(point_cloud.points).astype(np.float32)
        self.normals = np.asarray(point_cloud.normals).astype(np.float32)

        kd_tree = cKDTree(self.points)
        dist, _ = kd_tree.query(self.points, k=51, workers=-1)
        sigmas = dist[:, -1:].astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas).to(config.device)
        self.points = torch.from_numpy(self.points).to(config.device)
        self.normals = torch.from_numpy(self.normals).to(config.device)

    def sample(self) -> TrainingData:
        n = len(self.points)
        n_points = self.config.surface_points_count
        device = self.config.device

        manifold_indices = torch.randperm(n)
        manifold_idx = manifold_indices[:n_points]
        manifold_points = self.points[manifold_idx]
        manifold_normals = self.normals[manifold_idx]

        right = 0.5 + self.config.padding
        left = -right

        nonmanifold_points = (left - right) * torch.rand(size=(n_points, 3), dtype=torch.float32, device=device) + right
        gaussian_noise = torch.randn(size=manifold_points.shape, dtype=torch.float32, device=device)
        near_points = manifold_points + self.sigmas[manifold_idx] * gaussian_noise

        manifold_points.requires_grad_()
        manifold_normals.requires_grad_()
        nonmanifold_points.requires_grad_()
        near_points.requires_grad_()

        return TrainingData(manifold_points, manifold_normals, nonmanifold_points, near_points)
