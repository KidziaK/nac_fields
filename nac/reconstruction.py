import mcubes
import torch
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from nac import VoronoiNetwork


def reconstruct_mesh(
        networks: list[VoronoiNetwork] | VoronoiNetwork,
        resolution: int = 256,
        batch_size: int = 65536,
) -> o3d.geometry.TriangleMesh:
    if not isinstance(networks, list):
        networks = [networks]

    device = next(networks[0].parameters()).device

    x = np.linspace(-0.51, 0.51, resolution, dtype=np.float32)
    y = np.linspace(-0.51, 0.51, resolution, dtype=np.float32)
    z = np.linspace(-0.51, 0.51, resolution, dtype=np.float32)

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    grid_np = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    grid = torch.tensor(grid_np, dtype=torch.float32, device=device)

    all_sdf_values = torch.zeros(grid.shape[0], device=device, dtype=torch.uint8)
    udf = 100 * torch.ones(grid.shape[0], device=device, dtype=torch.float32)

    with torch.no_grad():
        for i in range(0, grid.shape[0], batch_size):
            batch_points = torch.tensor(grid[i:i + batch_size], dtype=torch.float32, device=device)
            current_slice = slice(i, i + batch_size)

            for nn in networks:
                nn.eval()
                sdf = nn(batch_points)
                all_sdf_values[current_slice] += torch.where(sdf < 0, 1, 0).squeeze()
                udf[current_slice] = torch.minimum(udf[current_slice], torch.abs(sdf.squeeze()))

    sdf_volume = all_sdf_values.reshape(resolution, resolution, resolution)
    udf_volume = udf.reshape(resolution, resolution, resolution)
    sdf_volume = torch.where(sdf_volume == sdf_volume.max(), -udf_volume, udf_volume)

    sdf_np = sdf_volume.cpu().numpy()

    vertices, faces = mcubes.marching_cubes(sdf_np, 0)

    mesh = o3d.geometry.TriangleMesh()

    a = 0.50
    bbox = np.array([[-a, a], [-a, a], [-a, a]])
    bbox_size = bbox[:, 1] - bbox[:, 0]
    vertices = vertices * (bbox_size / resolution)
    vertices = vertices + bbox[:, 0]

    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    mesh.compute_vertex_normals()

    return mesh


def prune(mesh: o3d.geometry.TriangleMesh, pcd: o3d.geometry.PointCloud, k: int = 5) -> o3d.geometry.TriangleMesh:
    pcd_points = np.asarray(pcd.points)
    kdtree = cKDTree(pcd_points)

    closest_pcd_dist, _ = kdtree.query(pcd_points, k=k)
    max_neighbor_dist = np.max(closest_pcd_dist)

    vertices = np.asarray(mesh.vertices)
    distances, _ = kdtree.query(vertices, k=k)
    mean_distances = np.max(distances, axis=1)

    vertices_to_remove_mask = mean_distances > max_neighbor_dist
    vertex_indices_to_remove = np.where(vertices_to_remove_mask)[0]

    if vertex_indices_to_remove.size > 0:
        triangles = np.asarray(mesh.triangles)
        triangles_to_remove_mask = np.any(np.isin(triangles, vertex_indices_to_remove), axis=1)
        mesh.remove_triangles_by_mask(triangles_to_remove_mask)
        mesh.remove_unreferenced_vertices()

    return mesh
