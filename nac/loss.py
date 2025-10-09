import torch
import numpy as np
import torch.nn.functional as F
from enum import auto, Enum
from scipy import spatial
from torch import Tensor
from torch import mean, abs, exp, norm


class SampleDirection(Enum):
    POSITIVE = auto()
    NEGATIVE = auto()
    BOTH = auto()


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


def orientation_loss(query_points: Tensor, on_manifold_points: Tensor, on_manifold_normals: Tensor) -> Tensor:
    orientation_signs = compute_orientation_sign(query_points, on_manifold_points, on_manifold_normals)
    target_signs = orientation_signs.unsqueeze(-1)
    return torch.mean(torch.relu(-query_points * target_signs))


def manifold_loss(x: Tensor) -> Tensor:
    integrand = abs(x)
    return mean(integrand)


def non_manifold_loss(x: Tensor, alpha: float) -> Tensor:
    integrand = exp(-alpha * abs(x))
    return mean(integrand)


def eikonal_loss(dx: Tensor) -> Tensor:
    integrand = abs(norm(dx, dim=1) - 1)
    return mean(integrand)


def optimized_finite_proxy_loss(x, y, u, v, h_step, model):
    """
    Optimized version that reduces network calls from 6 to 4
    by using a more efficient finite difference stencil
    """

    def f(pts: torch.Tensor):
        return model(pts).view(-1)

    # Original approach uses 6 calls - we can reduce to 4
    # Using forward differences instead of central differences
    f00 = y.view(-1)

    # Only compute the 4 corner points instead of 6 points
    f_u = f(x + h_step * u)  # Call 1
    f_v = f(x + h_step * v)  # Call 2
    f_uv = f(x + h_step * (u + v))  # Call 3
    f_origin = f(x)  # Call 4 (could reuse f00)

    # Compute mixed derivative using forward differences
    Duv = (f_uv - f_u - f_v + f_origin) / (h_step * h_step)
    return Duv.abs()


def get_fermi(model, points, grad, offset_d):
    x = points.detach().requires_grad_(True)
    n = F.normalize(grad, p=2, dim=-1, eps=1e-9)
    n.detach()
    rand = torch.randn(n.shape, device=x.device)
    x_u = F.normalize(torch.cross(n, rand, dim=-1), dim=-1)
    x_v = F.normalize(torch.cross(n, x_u, dim=-1), dim=-1)
    x_w = offset_d * n + x
    return x_u.detach(), x_v.detach(), x_w.detach()


def optimized_get_proxy_loss(model, points, offset_d, h_step, finite_difference, optimization_level, config):
    pred = model(points)
    grad = compute_gradient(points, pred)
    u, v, x_w = get_fermi(model, points, grad, offset_d)
    x_w = x_w.requires_grad_(True)
    pred_w = model(x_w)
    grad_w = compute_gradient(x_w, pred_w)

    loss = optimized_finite_proxy_loss(x_w, pred_w, u, v, h_step, model)

    return loss.mean()