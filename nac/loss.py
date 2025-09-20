import torch
import numpy as np
from typing import Dict
from scipy import spatial
from torch import nn
from .data import TrainingData
from torch import Tensor


def compute_gradient(inputs: Tensor, outputs: Tensor) -> Tensor:
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return points_grad


def compute_orientation_sign(query_points: Tensor, surface_points: Tensor, surface_normals: Tensor) -> Tensor:
    query_np = query_points.detach().cpu().numpy()
    surface_np = surface_points.detach().cpu().numpy()
    normals_np = surface_normals.detach().cpu().numpy()

    kd_tree = spatial.KDTree(surface_np)
    distances, indices = kd_tree.query(query_np, k=2, workers=-1)

    nearest_surface_points = surface_np[indices]
    nearest_normals = normals_np[indices]

    vectors_to_query = query_np[:, np.newaxis, :] - nearest_surface_points

    dot_products = np.sum(vectors_to_query * nearest_normals, axis=2)

    avg_dot_product = np.mean(dot_products, axis=1)

    orientation_signs_np = np.sign(avg_dot_product)

    orientation_signs = torch.tensor(orientation_signs_np, device=query_points.device, dtype=torch.float32)

    return orientation_signs


class NormalBasedSDFLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.sdf_weight = 7e3
        self.eikonal_weight = 6e2
        self.orientation_weight = 5e2
        self.near_surface_orientation_weight = 10
        self.gradient_normal_weight = 2e2


    def forward(self, output_pred: Dict[str, Tensor], training_data: TrainingData) -> Dict[str, Tensor]:
        manifold_pred = output_pred["manifold_pnts_pred"]
        nonmanifold_pred = output_pred["nonmanifold_pnts_pred"]
        near_points_pred = output_pred.get("near_points_pred")

        manifold_points = training_data.surface_points
        manifold_normals = training_data.surface_normals
        nonmanifold_points = training_data.off_surface_points
        near_points = training_data.near_surface_points

        manifold_points.requires_grad_()
        nonmanifold_points.requires_grad_()
        near_points.requires_grad_()

        manifold_grad = compute_gradient(manifold_points, manifold_pred)

        sdf_loss = torch.mean(manifold_pred ** 2)
        eikonal_loss = torch.mean((torch.norm(manifold_grad, dim=-1) - 1) ** 2)

        orientation_signs = compute_orientation_sign(nonmanifold_points, manifold_points, manifold_normals)
        target_signs = orientation_signs.unsqueeze(-1)
        orientation_loss = torch.mean(torch.relu(-nonmanifold_pred * target_signs))

        orientation_signs = compute_orientation_sign(near_points, manifold_points, manifold_normals)
        target_signs = orientation_signs.unsqueeze(-1)
        near_surface_orientation_loss = torch.mean(torch.relu(-near_points_pred * target_signs))

        gradient_normal_loss = torch.mean((manifold_grad - manifold_normals) ** 2)

        total_loss = (self.sdf_weight * sdf_loss +
                      self.eikonal_weight * eikonal_loss +
                      self.orientation_weight * orientation_loss +
                      self.near_surface_orientation_weight * near_surface_orientation_loss +
                      self.gradient_normal_weight * gradient_normal_loss)

        loss_dict = {
            'loss': total_loss,
            'sdf_loss': sdf_loss,
            'eikonal_loss': eikonal_loss,
            'orientation_loss': orientation_loss,
            'near_surface_orientation_loss': near_surface_orientation_loss,
            'gradient_normal_loss': gradient_normal_loss
        }

        return loss_dict
