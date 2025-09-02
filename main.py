import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import trimesh
import open3d as o3d
import os
from tqdm import tqdm
import mcubes
from typing import Dict, Tuple
import logging
import scipy.spatial as spatial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Sine(nn.Module):
    """Sine activation function for SIREN architecture."""

    def forward(self, input):
        return torch.sin(30 * input)


def sine_init(m):
    """Initialize weights for SIREN architecture."""
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    """Initialize first layer weights for SIREN architecture."""
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class FCBlock(nn.Module):
    """Fully connected block for SIREN architecture."""

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='sine', init_type='siren'):
        super().__init__()

        self.init_type = init_type

        nl_dict = {'sine': Sine(), 'relu': nn.ReLU(inplace=True)}
        nl = nl_dict[nonlinearity]

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
    """Self-contained absolute value layer for UDF."""

    def __init__(self):
        super(AbsLayer, self).__init__()

    def forward(self, x):
        return torch.abs(x)


class Decoder(nn.Module):
    """Decoder for SIREN architecture."""

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
    """NeurCADRecon network implementation."""

    def __init__(self, in_dim=3, decoder_hidden_dim=256, decoder_n_hidden_layers=4,
                 init_type='siren', sphere_init_params=[1.6, 0.1], udf=False):
        super().__init__()

        self.init_type = init_type
        self.decoder = Decoder(udf=udf)
        self.decoder.fc_block = FCBlock(
            in_dim, 1,
            num_hidden_layers=decoder_n_hidden_layers,
            hidden_features=decoder_hidden_dim,
            outermost_linear=True,
            nonlinearity='sine',
            init_type=init_type
        )

    def forward(self, points):
        """Forward pass through the network."""
        return self.decoder(points)


class MeshPreprocessor:
    """Handles mesh loading, preprocessing, and sampling."""

    def __init__(self, mesh_path: str):
        self.mesh_path = mesh_path
        self.mesh = None
        self.points = None
        self.normals = None
        self.center = None
        self.scale = None
        self.bbox = None

    def load_and_preprocess(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load mesh and preprocess it."""
        logger.info(f"Loading mesh from {self.mesh_path}")

        self.mesh = trimesh.load(self.mesh_path, process=False)
        if isinstance(self.mesh, trimesh.Scene):
            self.mesh = trimesh.load(self.mesh_path, process=False, force='mesh')

        points, face_indices = self.mesh.sample(10000, return_index=True)
        normals = self.mesh.face_normals[face_indices]

        self.center = points.mean(axis=0)
        points = points - self.center[None, :]
        self.scale = np.abs(points).max()
        points = points / (self.scale * 2.0)

        self.bbox = np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]])

        self.points = points.astype(np.float32)
        self.normals = normals.astype(np.float32)

        self.estimated_normals = self.estimate_normals_with_open3d(self.points)

        logger.info(f"Preprocessed mesh: {self.points.shape[0]} points, scale: {self.scale:.4f}")
        return self.points, self.normals

    def estimate_normals_with_open3d(self, points: np.ndarray, k_neighbors: int = 30) -> np.ndarray:
        """Estimate normals from point cloud using Open3D's built-in method."""
        logger.info(f"Estimating normals from {points.shape[0]} points using Open3D")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors)
        )

        estimated_normals = np.asarray(pcd.normals)

        logger.info("Open3D normal estimation completed")
        return estimated_normals.astype(np.float32)

    def sample_training_data(self, n_points: int = 20000, n_samples: int = 10000, use_estimated_normals: bool = True) -> \
    Dict[str, np.ndarray]:
        """Sample training data including manifold and non-manifold points."""

        manifold_indices = np.random.permutation(self.points.shape[0])
        manifold_idx = manifold_indices[:n_points]
        manifold_points = self.points[manifold_idx]

        if use_estimated_normals:
            manifold_normals = self.estimated_normals[manifold_idx]
        else:
            manifold_normals = self.normals[manifold_idx]

        nonmanifold_points = np.random.uniform(-0.5, 0.5, size=(n_points, 3)).astype(np.float32)

        kd_tree = spatial.KDTree(self.points)
        dist, _ = kd_tree.query(self.points, k=51, workers=-1)
        sigmas = dist[:, -1:]

        near_points = (manifold_points + sigmas[manifold_idx] *
                       np.random.randn(manifold_points.shape[0], manifold_points.shape[1])).astype(np.float32)

        return {
            'points': manifold_points,
            'mnfld_n': manifold_normals,
            'nonmnfld_points': nonmanifold_points,
            'near_points': near_points,
        }


class NormalBasedSDFLoss(nn.Module):
    """SDF loss that uses normals to determine inside/outside for non-watertight surfaces."""

    def __init__(self, loss_type='normal_based_sdf'):
        super().__init__()

        self.sdf_weight = 7e3
        self.eikonal_weight = 6e2
        self.normal_weight = 0
        self.orientation_weight = 5e1
        self.morse_weight = 10
        self.gradient_normal_weight = 2e2

        self.loss_type = loss_type
        self.use_morse = 'morse' in loss_type

    def gradient(self, inputs, outputs, create_graph=True, retain_graph=True):
        """Compute gradients."""
        d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
        points_grad = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=d_points,
            create_graph=create_graph,
            retain_graph=retain_graph,
            only_inputs=True)[0]
        return points_grad

    def compute_orientation_sign_efficient(self, query_points, surface_points, surface_normals, k_neighbors=1):
        """
        Compute the sign for query points using scipy KD-tree for efficiency.
        Positive sign means the point is on the 'inside' side of the surface.
        """
        query_np = query_points.detach().cpu().numpy()
        surface_np = surface_points.detach().cpu().numpy()
        normals_np = surface_normals.detach().cpu().numpy()

        kd_tree = spatial.KDTree(surface_np)

        distances, indices = kd_tree.query(query_np, k=k_neighbors, workers=-1)

        orientation_signs = []
        for i, query_point in enumerate(query_np):
            nearest_surface_points = surface_np[indices[i]]
            nearest_normals = normals_np[indices[i]]
            vectors_to_query = query_point - nearest_surface_points
            dot_products = np.sum(vectors_to_query * nearest_normals, axis=1)
            avg_dot_product = np.mean(dot_products)
            orientation_sign = np.sign(avg_dot_product)
            orientation_signs.append(orientation_sign)

        orientation_signs = torch.tensor(orientation_signs, device=query_points.device, dtype=torch.float32)
        return orientation_signs

    def compute_orientation_sign(self, query_points, surface_points, surface_normals, k_neighbors=2):
        """
        Compute the sign for query points based on their relationship to surface points and normals.
        Uses efficient KD-tree implementation for better performance.
        """
        return self.compute_orientation_sign_efficient(query_points, surface_points, surface_normals, k_neighbors)

    def forward(self, output_pred: Dict[str, torch.Tensor], gt: Dict[str, torch.Tensor]) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute normal-based SDF loss."""

        manifold_pred = output_pred["manifold_pnts_pred"]
        nonmanifold_pred = output_pred["nonmanifold_pnts_pred"]
        near_points_pred = output_pred.get("near_points_pred")

        manifold_points = gt['points']
        manifold_normals = gt['mnfld_n']
        nonmanifold_points = gt['nonmnfld_points']
        near_points = gt.get('near_points')

        manifold_points.requires_grad_()
        nonmanifold_points.requires_grad_()
        if near_points is not None:
            near_points.requires_grad_()

        manifold_grad = self.gradient(manifold_points, manifold_pred)
        nonmanifold_grad = self.gradient(nonmanifold_points, nonmanifold_pred)

        sdf_loss = torch.mean(manifold_pred ** 2)
        eikonal_loss = torch.mean((torch.norm(manifold_grad, dim=-1) - 1) ** 2)
        normal_loss = torch.mean((manifold_grad - manifold_normals) ** 2)

        orientation_signs = self.compute_orientation_sign(
            nonmanifold_points, manifold_points, manifold_normals, k_neighbors=2
        )
        target_signs = orientation_signs.unsqueeze(-1)
        orientation_loss = torch.mean(torch.relu(-nonmanifold_pred * target_signs + 0.1))

        morse_loss = torch.tensor(0.0, device=manifold_points.device)
        if self.use_morse and near_points_pred is not None:
            near_grad = self.gradient(near_points, near_points_pred)
            morse_loss = torch.mean(torch.relu(-near_points_pred + 0.05))

        gradient_normal_loss = torch.mean((manifold_grad - manifold_normals) ** 2)

        total_loss = (self.sdf_weight * sdf_loss +
                      self.eikonal_weight * eikonal_loss +
                      self.normal_weight * normal_loss +
                      self.orientation_weight * orientation_loss +
                      self.morse_weight * morse_loss +
                      self.gradient_normal_weight * gradient_normal_loss)

        loss_dict = {
            'loss': total_loss,
            'sdf_loss': sdf_loss,
            'eikonal_loss': eikonal_loss,
            'normal_loss': normal_loss,
            'orientation_loss': orientation_loss,
            'morse_loss': morse_loss,
            'gradient_normal_loss': gradient_normal_loss
        }

        return total_loss, loss_dict


class MeshReconstructor:
    """Handles mesh reconstruction from trained network."""

    def __init__(self, network: NeurCADReconNetwork, device: str = 'cuda'):
        self.network = network
        self.device = device

    def get_3d_grid(self, resolution: int = 256, bbox: np.ndarray = None) -> Dict[str, torch.Tensor]:
        """Generate 3D grid for marching cubes."""
        if bbox is None:
            bbox = np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]])

        x = np.linspace(bbox[0, 0], bbox[0, 1], resolution)
        y = np.linspace(bbox[1, 0], bbox[1, 1], resolution)
        z = np.linspace(bbox[2, 0], bbox[2, 1], resolution)

        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T,
                                   dtype=torch.float32, device=self.device)

        return {
            "grid_points": grid_points,
            "xyz": [x, y, z]
        }

    def reconstruct_mesh(self, resolution: int = 256, chunk_size: int = 10000) -> trimesh.Trimesh:
        """Reconstruct mesh using marching cubes for SDF."""
        logger.info("Starting mesh reconstruction for SDF...")

        self.network.eval()
        grid_dict = self.get_3d_grid(resolution=resolution)
        grid_points = grid_dict["grid_points"]

        z_values = []
        with torch.no_grad():
            for i in tqdm(range(0, grid_points.shape[0], chunk_size), desc="Evaluating SDF"):
                chunk = grid_points[i:i + chunk_size]
                chunk_pred = self.network(chunk)
                z_values.append(chunk_pred.cpu().numpy())

        z_values = np.concatenate(z_values, axis=0)

        x, y, z = grid_dict["xyz"]
        z_grid = z_values.reshape(len(x), len(y), len(z))

        logger.info("Applying marching cubes with threshold 0...")
        vertices, faces = mcubes.marching_cubes(z_grid, 0)

        vertices = vertices * (1.0 / resolution) * 1.0

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        logger.info(f"Reconstructed mesh: {len(vertices)} vertices, {len(faces)} faces")
        return mesh


def train_neurcadrecon(mesh_path: str, output_dir: str, num_epochs: int = 100,
                       batch_size: int = 1, lr: float = 5e-5, device: str = 'cuda'):
    """Main training function for normal-based SDF reconstruction."""

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info("Training with normal-based SDF for non-watertight surfaces")

    preprocessor = MeshPreprocessor(mesh_path)
    points, normals = preprocessor.load_and_preprocess()

    network = NeurCADReconNetwork(
        in_dim=3,
        decoder_hidden_dim=256,
        decoder_n_hidden_layers=4,
        init_type='siren',
        udf=False
    ).to(device)

    optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=0.0)
    criterion = NormalBasedSDFLoss()

    logger.info("Starting training...")
    best_loss = float('inf')

    for epoch in range(num_epochs):
        network.train()

        training_data = preprocessor.sample_training_data(n_points=20000, n_samples=1, use_estimated_normals=True)

        for key in training_data:
            training_data[key] = torch.tensor(training_data[key], device=device, dtype=torch.float32)
            training_data[key].requires_grad_()

        manifold_pred = network(training_data['points'])
        nonmanifold_pred = network(training_data['nonmnfld_points'])
        near_points_pred = network(training_data['near_points'])

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
            logger.info(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.6f}")
            logger.info(f"  SDF: {loss_dict['sdf_loss']:.6f}, Eikonal: {loss_dict['eikonal_loss']:.6f}")
            logger.info(f"  Normal: {loss_dict['normal_loss']:.6f}, Orientation: {loss_dict['orientation_loss']:.6f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(network.state_dict(), os.path.join(output_dir, 'best_model.pth'))

    logger.info(f"Training completed. Best loss: {best_loss:.6f}")

    logger.info("Reconstructing mesh using normal-based SDF...")
    reconstructor = MeshReconstructor(network, device)
    reconstructed_mesh = reconstructor.reconstruct_mesh(resolution=256)

    output_mesh_path = os.path.join(output_dir, 'reconstructed_mesh.obj')
    reconstructed_mesh.export(output_mesh_path)
    logger.info(f"Reconstructed mesh saved to: {output_mesh_path}")

    return network, reconstructed_mesh


def main():
    input_mesh_path = "/home/mikolaj/Documents/github/inr_voronoi/data/sphylinder/sphylinder.obj"
    output_directory = "/home/mikolaj/Downloads"
    num_epochs = 200
    batch_size = 1
    learning_rate = 5e-5
    device = 'cuda'

    if not os.path.exists(input_mesh_path):
        logger.error(f"Input mesh file not found: {input_mesh_path}")
        return

    try:
        network, mesh = train_neurcadrecon(
            mesh_path=input_mesh_path,
            output_dir=output_directory,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=learning_rate,
            device=device
        )
        logger.info("UDF-based mesh reconstruction completed successfully!")

    except Exception as e:
        logger.error(f"Error during reconstruction: {e}")
        raise


if __name__ == "__main__":
    main()
