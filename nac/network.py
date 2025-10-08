import torch
import numpy as np
from torch import nn, optim
from tqdm import tqdm

from .data import SirenDataset
from .settings import TrainingConfig
from .loss import manifold_loss, non_manifold_loss, eikonal_loss, first_order_morse_loss
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
        self.morse_weight = 10

    def train_point_cloud(self, config: TrainingConfig, dataset: SirenDataset):
        optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        self.train()
        self.to(config.device)

        progress_bar = tqdm(range(config.epochs))
        for _ in progress_bar:
            data = dataset.sample()
            on_manifold_points = data.on_manifold_points
            off_manifold_points = data.off_manifold_points

            on_manifold_sdf = self(on_manifold_points)
            off_manifold_sdf = self(off_manifold_points)

            manifold_grad = compute_gradient(on_manifold_points, on_manifold_sdf)

            on_manifold_term = manifold_loss(on_manifold_sdf)
            off_manifold_term = non_manifold_loss(off_manifold_sdf, alpha=1e2)
            eikonal_term = eikonal_loss(manifold_grad)
            morse_term = first_order_morse_loss(self, on_manifold_points)

            total_loss = (self.manifold_weight * on_manifold_term +
                          self.non_manifold_weight * off_manifold_term +
                          self.eikonal_weight * eikonal_term +
                          self.morse_weight * morse_term)

            loss_dict = {
                'loss': total_loss,
                'manifold_term': on_manifold_term,
                'non_manifod_term': off_manifold_term,
                'eikonal_term': eikonal_term,
                'morse_term': morse_term,
            }

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
            optimizer.step()

            progress_bar.set_postfix({loss_name: loss_value.item() for loss_name, loss_value in loss_dict.items()})
