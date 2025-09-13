import mcubes
import torch
from nac import VoronoiNetwork
import numpy as np
import open3d as o3d

def reconstruct_mesh(networks: list[VoronoiNetwork] | VoronoiNetwork, resolution: int = 256, batch_size: int = 65536):
    if not isinstance(networks, list):
        networks = [networks]

    device = next(networks[0].parameters()).device

    x = np.linspace(-0.51, 0.51, resolution, dtype=np.float32)
    y = np.linspace(-0.51, 0.51, resolution, dtype=np.float32)
    z = np.linspace(-0.51, 0.51, resolution, dtype=np.float32)

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    grid_np = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    grid = torch.tensor(grid_np, dtype=torch.float32, device=device)

    all_sdf_values = torch.zeros(grid.shape[0], device=device)

    with torch.no_grad():
        for i in range(0, grid.shape[0], batch_size):
            batch_points = torch.tensor(grid[i:i + batch_size], dtype=torch.float32, device=device)

            all_batch_sdfs = []
            for nn in networks:
                nn.eval()
                sdf = nn(batch_points)
                all_batch_sdfs.append(sdf)

            stacked_sdfs = torch.stack(all_batch_sdfs)
            min_sdf, _ = torch.min(stacked_sdfs, dim=0)
            all_sdf_values[i:i + batch_size] = min_sdf.squeeze()

    sdf_volume = all_sdf_values.reshape(resolution, resolution, resolution)

    sdf_np = sdf_volume.cpu().numpy()

    vertices, faces = mcubes.marching_cubes(sdf_np, 0)

    mesh = o3d.geometry.TriangleMesh()

    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    mesh.compute_vertex_normals()

    return mesh
