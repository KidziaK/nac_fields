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


def manifold_loss(x: Tensor) -> Tensor:
    integrand = abs(x)
    return mean(integrand)


def non_manifold_loss(x: Tensor, alpha: float) -> Tensor:
    integrand = exp(-alpha * abs(x))
    return mean(integrand)


def eikonal_loss(dx: Tensor) -> Tensor:
    integrand = (norm(dx, dim=1) - 1) ** 2
    return mean(integrand)


def sample_shell(
        model,
        x_surf: torch.Tensor,
        d_offset: float,
        d_direction: SampleDirection = SampleDirection.BOTH,
        n_shell: int | None = None,
):
    if n_shell is None:
        n_shell = x_surf.shape[0]

    # ------------- pick random subset and enable grad for ∇f -------------
    idx = torch.randperm(x_surf.size(0), device=x_surf.device)[:n_shell]
    x_s = x_surf[idx].detach().requires_grad_(True)  # (M,3)

    # ------------- compute unit normals n = ∇f / ‖∇f‖ --------------------
    fval = model(x_s).sum()
    n = torch.autograd.grad(fval, x_s, create_graph=True)[0]
    n = F.normalize(n, dim=-1, eps=1e-9)

    # ------------- random orthonormal tangents u, v ----------------------
    rand = torch.randn_like(n)
    u = F.normalize(rand - (rand * n).sum(-1, keepdim=True) * n, dim=-1)
    v = F.normalize(torch.cross(n, u, dim=-1), dim=-1)

    # ------------- shell point  x + d n  with  d ∈ (0,d_offset] ----------
    match d_direction:
        case SampleDirection.POSITIVE:
            d = torch.empty(n_shell, 1, device=x_s.device).uniform_(0.0, d_offset)
        case SampleDirection.NEGATIVE:
            d = -torch.empty(n_shell, 1, device=x_s.device).uniform_(0.0, d_offset)
        case SampleDirection.BOTH:
            d = torch.empty(n_shell, 1, device=x_s.device).uniform_(-d_offset, d_offset)
        case _:
            raise ValueError(f"Invalid d_direction: {d_direction}. Choose from 'positive', 'negative', or 'both'.")

    x_w = x_s + d * n # (M,3)

    return x_w.detach(), u.detach(), v.detach()


def first_order_morse_loss(
        model,
        x_surf: torch.Tensor,
        d_offset: float = 0.05,
        h_step: float = 0.02,
        central: bool = False,
        d_direction: SampleDirection = SampleDirection.BOTH,
        n_shell: int | None = None,
):
    if h_step is None:
        h_step = d_offset

    # ---------- sample shell Ω_t (normal offset only uses d_offset) -------
    x_w, u, v = sample_shell(model, x_surf, d_offset, d_direction=d_direction, n_shell=n_shell)
    M = x_w.shape[0]

    # ---------- helper that enforces finite model output ------------------
    def f(pts: torch.Tensor):
        y = model(pts.view(-1, 3)).reshape(-1)
        return y

    # ---------- one-sided mixed stencil using step size h -----------------
    f00 = f(x_w)
    fu = f(x_w + h_step * u)
    fv = f(x_w + h_step * v)
    fuv = f(x_w + h_step * (u + v))

    Duv = (fuv - fu - fv + f00) / (h_step * h_step)
    loss = Duv.abs()  # (M,)

    # ---------- optional central (±h) correction --------------------------
    if central:
        fu_m = f(x_w - h_step * u)
        fv_m = f(x_w - h_step * v)
        fuv_m = f(x_w - h_step * (u + v))
        Duv_m = (fuv_m - fu_m - fv_m + f00) / (h_step * h_step)
        loss = 0.5 * (Duv + Duv_m).abs()

    return loss.mean()
