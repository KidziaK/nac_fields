import mcubes
import torch
import numpy as np
import open3d as o3d
from nac.network import Siren
from nac.settings import ReconstructionConfig


def reconstruct_mesh(network: Siren, config: ReconstructionConfig) -> o3d.geometry.TriangleMesh:
    device = config.device
    resolution = config.grid_resolution
    extent = config.bounding_box_extent

    x = np.linspace(-extent, extent, resolution, dtype=np.float32)
    y = np.linspace(-extent, extent, resolution, dtype=np.float32)
    z = np.linspace(-extent, extent, resolution, dtype=np.float32)

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    grid_np = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    grid = torch.tensor(grid_np, dtype=torch.float32, device=device)

    sdf_flat = torch.zeros(resolution ** 3, dtype=torch.float32, device=device).unsqueeze(-1)

    with torch.no_grad():
        network.to(device)
        network.eval()
        batch_size = 2 ** 20

        for idx in range(0, len(grid), batch_size):
            current_slice = slice(idx, idx + batch_size)
            sdf_flat[current_slice] = network(grid[current_slice])

    sdf = sdf_flat.reshape(resolution, resolution, resolution)
    sdf_np = sdf.cpu().numpy()

    vertices, faces = mcubes.marching_cubes(sdf_np, -config.offset)

    mesh = o3d.geometry.TriangleMesh()

    vertices = vertices * 2 * extent / resolution

    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    mesh.compute_vertex_normals()

    return mesh
