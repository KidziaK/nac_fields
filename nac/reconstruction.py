import mcubes
import torch
import numpy as np
import open3d as o3d
from scipy.ndimage import zoom
from nac import VoronoiNetwork


def transform_mesh_to_aabb(
        mesh: o3d.geometry.TriangleMesh,
        new_aabb: o3d.geometry.AxisAlignedBoundingBox
) -> o3d.geometry.TriangleMesh:
    original_aabb = mesh.get_axis_aligned_bounding_box()
    original_min = original_aabb.min_bound
    original_max = original_aabb.max_bound
    original_extents = original_max - original_min

    new_min = new_aabb.min_bound
    new_max = new_aabb.max_bound
    new_extents = new_max - new_min

    scale_factor = new_extents / original_extents

    translation_vector = new_min - original_min

    T = np.identity(4)
    S = np.identity(4)
    S[:3, :3] = np.diag(scale_factor)
    T[:3, 3] = translation_vector

    transform_matrix = T @ S

    transformed_mesh = mesh.transform(transform_matrix)

    return transformed_mesh


def reconstruct_mesh(
        networks: list[VoronoiNetwork] | VoronoiNetwork,
        resolution: int = 256,
        batch_size: int = 65536,
        aabb: o3d.geometry.AxisAlignedBoundingBox | None = None
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

    zoom_factor = 1
    vertices, faces = mcubes.marching_cubes(zoom(sdf_np, zoom_factor, order=3), 0)

    mesh = o3d.geometry.TriangleMesh()

    a = 0.51
    bbox = np.array([[-a, a], [-a, a], [-a, a]])
    bbox_size = bbox[:, 1] - bbox[:, 0]
    vertices = vertices * (bbox_size / resolution) / zoom_factor
    vertices = vertices + bbox[:, 0]

    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    if aabb:
        mesh = transform_mesh_to_aabb(mesh, aabb)

    mesh.compute_vertex_normals()

    return mesh
