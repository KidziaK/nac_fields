import torch
import numpy as np
from torch import nn, optim
from tqdm import tqdm

from .data import SirenDataset
from .settings import TrainingConfig
from .loss import manifold_loss, non_manifold_loss, eikonal_loss, optimized_get_proxy_loss, orientation_loss
from abc import ABC, abstractmethod
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


class Sine(nn.Module):
    def forward(self, input):
        return torch.sin(30 * input)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class Siren(nn.Module, ABC):
    def __init__(self, in_features: int = 3, out_features: int = 1, num_hidden_layers: int = 4, hidden_features: int = 256):
        super().__init__()

        nl = Sine()

        self.net = []
        self.net.append(nn.Sequential(nn.Linear(in_features, hidden_features), nl))

        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(nn.Linear(hidden_features, hidden_features), nl))

        self.net.append(nn.Sequential(nn.Linear(hidden_features, out_features)))

        self.net = nn.Sequential(*self.net)

        self.net.apply(sine_init)
        self.net[0].apply(first_layer_sine_init)

    def forward(self, coords):
        return self.net(coords)

    @abstractmethod
    def train_point_cloud(self, config: TrainingConfig, dataset: SirenDataset):
        raise NotImplementedError()


class FlatCAD(Siren):
    def __init__(self):
        super().__init__()

        self.manifold_weight = 7e3
        self.non_manifold_weight = 6e2
        self.eikonal_weight = 5e1
        self.normal_weight = 2e2
        self.orientation_weight = 5e2

    def train_point_cloud(self, config: TrainingConfig, dataset: SirenDataset):
        optimizer = optim.Adam(self.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.train()
        self.to(config.device)

        progress_bar = tqdm(range(config.epochs))
        for _ in progress_bar:
            data = dataset.sample()
            on_manifold_points = data.on_manifold_points
            off_manifold_points = data.off_manifold_points
            near_manifold_points = data.near_manifold_points
            on_manifold_normals = data.on_manifold_normals
            original_on_manifold_points = data.original_on_manifold_points

            on_manifold_sdf = self(on_manifold_points)
            off_manifold_sdf = self(off_manifold_points)
            near_manifold_sdf = self(near_manifold_points)
            original_on_manifold_sdf = self(original_on_manifold_points)

            on_manifold_grad = compute_gradient(on_manifold_points, on_manifold_sdf)
            off_manifold_grad = compute_gradient(off_manifold_points, off_manifold_sdf)
            near_manifold_grad = compute_gradient(near_manifold_points, near_manifold_sdf)

            on_manifold_term = manifold_loss(on_manifold_sdf) + manifold_loss(original_on_manifold_sdf + dataset.offset)
            off_manifold_term = non_manifold_loss(off_manifold_sdf, alpha=config.non_manifold_alpha)
            eikonal_term = eikonal_loss(on_manifold_grad) + eikonal_loss(off_manifold_grad) + eikonal_loss(near_manifold_grad)
            normal_term = manifold_loss(on_manifold_normals - on_manifold_grad)
            orientation_term = orientation_loss(off_manifold_points, on_manifold_points, on_manifold_normals)

            total_loss = (
                self.manifold_weight * on_manifold_term +
                self.non_manifold_weight * off_manifold_term +
                self.eikonal_weight * eikonal_term +
                self.normal_weight * normal_term +
                self.orientation_weight * orientation_term
            )

            loss_dict = {
                'loss': total_loss,
                'manifold_term': on_manifold_term,
                'non_manifod_term': off_manifold_term,
                'eikonal_term': eikonal_term,
                'normal_term': normal_term,
                'orientation_term': orientation_term,
            }

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), config.gradient_clip)
            optimizer.step()

            progress_bar.set_postfix({loss_name: loss_value.item() for loss_name, loss_value in loss_dict.items()})
