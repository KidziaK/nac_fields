import torch
import numpy as np
from torch import nn, optim
from tqdm import tqdm
from .data import TrainingConfig, DataSampler
from .loss import compute_orientation_sign, manifold_loss, non_manifold_loss, eikonal_loss, first_order_morse_loss
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
    def train_point_cloud(self, config: TrainingConfig, data_sampler: DataSampler):
        raise NotImplementedError()


class VoronoiNetwork(Siren):
    def __init__(self):
        super().__init__()

        self.sdf_weight = 7e3
        self.eikonal_weight = 6e2
        self.orientation_weight = 5e2
        self.near_surface_orientation_weight = 10
        self.gradient_normal_weight = 2e2

    def train_point_cloud(self, config: TrainingConfig, data_sampler: DataSampler):
        optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        progress_bar = tqdm(range(config.epochs))

        for _ in progress_bar:
            self.train()

            training_data = data_sampler.sample()

            manifold_points = training_data.surface_points
            manifold_normals = training_data.surface_normals
            nonmanifold_points = training_data.off_surface_points
            near_points = training_data.near_surface_points

            manifold_pred = self(training_data.surface_points)
            nonmanifold_pred = self(training_data.off_surface_points)
            near_points_pred = self(training_data.near_surface_points)

            manifold_grad = compute_gradient(training_data.surface_points, manifold_pred)

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

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
            optimizer.step()

            progress_bar.set_postfix({loss_name: loss_value.item() for loss_name, loss_value in loss_dict.items()})


class FlatCAD(Siren):
    def __init__(self):
        super().__init__()

        self.manifold_weight = 7e3
        self.non_manifold_weight = 6e2
        self.eikonal_weight = 5e1
        self.morse_weight = 10

    def train_point_cloud(self, config: TrainingConfig, data_sampler: DataSampler):
        optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        progress_bar = tqdm(range(config.epochs))

        for _ in progress_bar:
            self.train()

            training_data = data_sampler.sample()
            manifold_points = training_data.surface_points
            non_manifold_points = training_data.off_surface_points

            manifold_pred = self(training_data.surface_points)
            near_points_pred = self(training_data.near_surface_points)

            manifold_grad = compute_gradient(training_data.surface_points, manifold_pred)

            manifold_term = manifold_loss(manifold_pred)
            non_manifod_term = non_manifold_loss(near_points_pred, alpha=1e2) + non_manifold_loss(non_manifold_points, alpha=1e2)
            eikonal_term = eikonal_loss(manifold_grad)
            morse_term = first_order_morse_loss(self, manifold_points)

            total_loss = (self.manifold_weight * manifold_term +
                          self.non_manifold_weight * non_manifod_term +
                          self.eikonal_weight * eikonal_term +
                          self.morse_weight * morse_term)

            loss_dict = {
                'loss': total_loss,
                'manifold_term': manifold_term,
                'non_manifod_term': non_manifod_term,
                'eikonal_term': eikonal_term,
                'morse_term': morse_term,
            }

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
            optimizer.step()

            progress_bar.set_postfix({loss_name: loss_value.item() for loss_name, loss_value in loss_dict.items()})
