import os
import time
import mcubes
import numpy as np
import open3d as o3d
import scipy.spatial as spatial
import torch
import torch.nn as nn
import torch.optim as optim
import trimesh
from tqdm import tqdm
from scipy.spatial import cKDTree
from dataclasses import dataclass
from typing import Dict, Tuple
from pathlib import Path


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result
    return wrapper


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, init_type='siren'):
        super().__init__()

        self.init_type = init_type
        nl = Sine()

        self.net = []
        self.net.append(nn.Sequential(nn.Linear(in_features, hidden_features), nl))

        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(nn.Linear(hidden_features, hidden_features), nl))

        if outermost_linear:
            self.net.append(nn.Sequential(nn.Linear(hidden_features, out_features)))
        else:
            self.net.append(nn.Sequential(nn.Linear(hidden_features, out_features), nl))

        self.net = nn.Sequential(*self.net)

        if init_type == 'siren':
            self.net.apply(sine_init)
            self.net[0].apply(first_layer_sine_init)

    def forward(self, coords):
        return self.net(coords)


class AbsLayer(nn.Module):
    def __init__(self):
        super(AbsLayer, self).__init__()

    def forward(self, x):
        return torch.abs(x)


class Decoder(nn.Module):
    def __init__(self, udf=False):
        super(Decoder, self).__init__()
        if udf:
            self.nl = AbsLayer()
        else:
            self.nl = nn.Identity()

    def forward(self, coords):
        res = self.fc_block(coords)
        res = self.nl(res)
        return res


class NeurCADReconNetwork(nn.Module):
    def __init__(self, in_dim=3, decoder_hidden_dim=256, decoder_n_hidden_layers=4, udf=False):
        super().__init__()

        self.decoder = Decoder(udf=udf)
        self.decoder.fc_block = FCBlock(
            in_dim, 1,
            num_hidden_layers=decoder_n_hidden_layers,
            hidden_features=decoder_hidden_dim,
            outermost_linear=True,
            init_type='siren'
        )

    def forward(self, points):
        return self.decoder(points)


@dataclass
class TrainingData:
    surface_points: torch.Tensor
    surface_normals: torch.Tensor
    off_surface_points: torch.Tensor
    near_surface_points: torch.Tensor

class MeshPreprocessor:
    def __init__(self, mesh_path: str, padding: float = 0.05):
        self.mesh_path = mesh_path
        self.padding = padding
        self.points = None
        self.normals = None
        self.sigmas = None
        self.scale = None
        self.center = None

    def load_and_preprocess(self) -> None:
        mesh = o3d.io.read_triangle_mesh(self.mesh_path)

        aabb = mesh.get_axis_aligned_bounding_box()
        center = aabb.get_center()
        self.center = center
        mesh.translate(-center)

        aabb_centered = mesh.get_axis_aligned_bounding_box()
        extent = aabb_centered.get_extent()

        max_dimension = np.max(extent)
        scale_factor = 1.0 / max_dimension
        mesh.scale(scale_factor, center=np.array([0.0, 0.0, 0.0]))
        self.scale = scale_factor

        pcd = mesh.sample_points_poisson_disk(number_of_points=20000)
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(5)

        points_np = np.asarray(pcd.points).astype(np.float32)

        kd_tree = cKDTree(points_np)
        dist, _ = kd_tree.query(points_np, k=51, workers=-1)
        sigmas = dist[:, -1:].astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas).to(DEVICE)
        self.points = torch.from_numpy(points_np).to(DEVICE)
        self.normals = torch.from_numpy(np.asarray(pcd.normals).astype(np.float32)).to(DEVICE)


    def sample_training_data(self, n_points: int = 10000) -> TrainingData:
        manifold_indices = torch.randperm(len(self.points))
        manifold_idx = manifold_indices[:n_points]
        manifold_points = self.points[manifold_idx]
        manifold_normals = self.normals[manifold_idx]

        right = 0.5 + self.padding
        left = -right

        nonmanifold_points = (left - right) * torch.rand(size=(n_points, 3), dtype=torch.float32, device=DEVICE) + right
        gaussian_noise = torch.randn(size=manifold_points.shape, dtype=torch.float32, device=DEVICE)
        near_points = manifold_points + self.sigmas[manifold_idx] * gaussian_noise

        manifold_points.requires_grad_()
        manifold_normals.requires_grad_()
        nonmanifold_points.requires_grad_()
        near_points.requires_grad_()

        return TrainingData(manifold_points, manifold_normals, nonmanifold_points, near_points)


class NormalBasedSDFLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.sdf_weight = 7e3
        self.eikonal_weight = 6e2
        self.orientation_weight = 5e2
        self.near_surface_orientation_weight = 10
        self.gradient_normal_weight = 2e2

    def gradient(self, inputs, outputs, create_graph=True, retain_graph=True):
        d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
        points_grad = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=d_points,
            create_graph=create_graph,
            retain_graph=retain_graph,
            only_inputs=True)[0]
        return points_grad

    def compute_orientation_sign(self, query_points, surface_points, surface_normals, k_neighbors=2):
        query_np = query_points.detach().cpu().numpy()
        surface_np = surface_points.detach().cpu().numpy()
        normals_np = surface_normals.detach().cpu().numpy()

        kd_tree = spatial.KDTree(surface_np)
        distances, indices = kd_tree.query(query_np, k=k_neighbors, workers=-1)

        nearest_surface_points = surface_np[indices]
        nearest_normals = normals_np[indices]

        vectors_to_query = query_np[:, np.newaxis, :] - nearest_surface_points

        dot_products = np.sum(vectors_to_query * nearest_normals, axis=2)

        avg_dot_product = np.mean(dot_products, axis=1)

        orientation_signs_np = np.sign(avg_dot_product)

        orientation_signs = torch.tensor(orientation_signs_np, device=query_points.device, dtype=torch.float32)

        return orientation_signs

    def forward(self, output_pred: Dict[str, torch.Tensor], gt: TrainingData) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:

        manifold_pred = output_pred["manifold_pnts_pred"]
        nonmanifold_pred = output_pred["nonmanifold_pnts_pred"]
        near_points_pred = output_pred.get("near_points_pred")

        manifold_points = gt.surface_points
        manifold_normals = gt.surface_normals
        nonmanifold_points = gt.off_surface_points
        near_points = gt.near_surface_points

        manifold_points.requires_grad_()
        nonmanifold_points.requires_grad_()
        near_points.requires_grad_()

        manifold_grad = self.gradient(manifold_points, manifold_pred)

        sdf_loss = torch.mean(manifold_pred ** 2)
        eikonal_loss = torch.mean((torch.norm(manifold_grad, dim=-1) - 1) ** 2)

        orientation_signs = self.compute_orientation_sign(
            nonmanifold_points, manifold_points, manifold_normals, k_neighbors=2
        )
        target_signs = orientation_signs.unsqueeze(-1)
        orientation_loss = torch.mean(torch.relu(-nonmanifold_pred * target_signs))

        orientation_signs = self.compute_orientation_sign(
            near_points, manifold_points, manifold_normals, k_neighbors=2
        )
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

        return total_loss, loss_dict


class MeshReconstructor:
    def __init__(self, network: NeurCADReconNetwork, scale: float, center):
        self.network = network
        self.scale = scale
        self.center = center

    def get_3d_grid(self, resolution: int = 256, padding: float = 0.05) -> Dict[str, torch.Tensor]:
        bbox = np.array([[-0.5 - padding, 0.5 + padding], 
                        [-0.5 - padding, 0.5 + padding], 
                        [-0.5 - padding, 0.5 + padding]])

        x = np.linspace(bbox[0, 0], bbox[0, 1], resolution)
        y = np.linspace(bbox[1, 0], bbox[1, 1], resolution)
        z = np.linspace(bbox[2, 0], bbox[2, 1], resolution)

        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T,
                                   dtype=torch.float32, device=DEVICE)

        return {
            "grid_points": grid_points,
            "xyz": [x, y, z],
            "bbox": bbox
        }

    def reconstruct_mesh(self, resolution: int = 256, chunk_size: int = 10000, padding: float = 0.1) -> trimesh.Trimesh:
        self.network.eval()
        grid_dict = self.get_3d_grid(resolution=resolution, padding=padding)
        grid_points = grid_dict["grid_points"]

        z_values = []
        with torch.no_grad():
            for i in tqdm(range(0, grid_points.shape[0], chunk_size), desc="Evaluating SDF"):
                chunk = grid_points[i:i + chunk_size]
                chunk_pred = self.network(chunk)
                z_values.append(chunk_pred.cpu().numpy())

        z_values = np.concatenate(z_values, axis=0)

        x, y, z = grid_dict["xyz"]
        bbox = grid_dict["bbox"]
        z_grid = z_values.reshape(len(x), len(y), len(z))

        vertices, faces = mcubes.marching_cubes(z_grid, 0)

        bbox_size = bbox[:, 1] - bbox[:, 0]
        vertices = vertices * (bbox_size / resolution)
        vertices = vertices + bbox[:, 0]
        vertices /= self.scale
        vertices += self.center

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        return mesh

@timeit
def train_neurcadrecon(mesh_path: str | Path, output_dir: str, num_epochs: int, lr: float):
    mesh_path = Path(mesh_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = MeshPreprocessor(mesh_path, padding=0.05)
    preprocessor.load_and_preprocess()

    network = NeurCADReconNetwork(
        in_dim=3,
        decoder_hidden_dim=256,
        decoder_n_hidden_layers=4,
        udf=False
    ).to(DEVICE)

    optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=0.0)
    criterion = NormalBasedSDFLoss()

    for epoch in range(num_epochs):
        network.train()

        training_data = preprocessor.sample_training_data(n_points=20000)

        manifold_pred = network(training_data.surface_points)
        nonmanifold_pred = network(training_data.off_surface_points)
        near_points_pred = network(training_data.near_surface_points)

        output_pred = {
            "manifold_pnts_pred": manifold_pred,
            "nonmanifold_pnts_pred": nonmanifold_pred,
            "near_points_pred": near_points_pred
        }

        loss, loss_dict = criterion(output_pred, training_data)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 10.0)
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.6f}")
            print(f"  SDF: {loss_dict['sdf_loss']:.6f}, Eikonal: {loss_dict['eikonal_loss']:.6f}")
            print(f"  Orientation: {loss_dict['orientation_loss']:.6f}")

    reconstructor = MeshReconstructor(network, preprocessor.scale, preprocessor.center)
    reconstructed_mesh = reconstructor.reconstruct_mesh(resolution=256, padding=0.05)

    output_mesh_path = output_dir / mesh_path.name
    reconstructed_mesh.export(output_mesh_path)

    return network, reconstructed_mesh


if __name__ == "__main__":
    input_meshes_paths = [
        "/home/mikolaj/Documents/github/inr_voronoi/data/sphylinder/hemisphere.obj",
        "/home/mikolaj/Documents/github/inr_voronoi/data/sphylinder/cylinder.obj",
        "/home/mikolaj/Documents/github/inr_voronoi/data/sphylinder/disk.obj"
    ]
    output_directory = "/home/mikolaj/Downloads"
    num_epochs = 10000
    learning_rate = 5e-5

    for input_path in tqdm(input_meshes_paths):
        train_neurcadrecon(
            mesh_path=input_path,
            output_dir=output_directory,
            num_epochs=num_epochs,
            lr=learning_rate
        )
