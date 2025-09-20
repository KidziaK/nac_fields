import torch
import numpy as np
from torch import nn, optim, Tensor
from tqdm import tqdm
from .data import TrainingConfig, DataSampler
from .loss import NormalBasedSDFLoss


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


class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features):
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


class VoronoiNetwork(nn.Module):
    def __init__(self, in_dim: int = 3, decoder_hidden_dim: int = 256, decoder_n_hidden_layers:int = 4):
        super().__init__()

        self.fc_block = FCBlock(
            in_dim,
            1,
            num_hidden_layers=decoder_n_hidden_layers,
            hidden_features=decoder_hidden_dim
        )

    def forward(self, points: Tensor) -> Tensor:
        return self.fc_block(points)

    def train_point_cloud(self, config: TrainingConfig, data_sampler: DataSampler):
        optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        criterion = NormalBasedSDFLoss()

        progress_bar = tqdm(range(config.epochs))

        for _ in progress_bar:
            self.train()

            training_data = data_sampler.sample()

            manifold_pred = self(training_data.surface_points)
            nonmanifold_pred = self(training_data.off_surface_points)
            near_points_pred = self(training_data.near_surface_points)

            output_pred = {
                "manifold_pnts_pred": manifold_pred,
                "nonmanifold_pnts_pred": nonmanifold_pred,
                "near_points_pred": near_points_pred
            }

            loss_dict = criterion(output_pred, training_data)
            loss = loss_dict["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
            optimizer.step()

            progress_bar.set_postfix({loss_name: loss_value.item() for loss_name, loss_value in loss_dict.items()})
