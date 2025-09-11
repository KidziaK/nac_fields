import numba
import numpy as np
import open3d as o3d
import torch

from typing import List, Tuple, Optional
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from scipy.optimize import least_squares


def find_connected_components(nodes, adjacency_matrix):
    graph = csr_matrix(adjacency_matrix)

    n_components, labels = connected_components(
        csgraph=graph, directed=False, return_labels=True
    )

    if n_components == 0:
        return []

    unique_labels, counts = np.unique(labels, return_counts=True)

    large_component_labels = unique_labels[counts > 50]

    result = [
        nodes[labels == label] for label in large_component_labels
    ]

    return result

@numba.njit(parallel=True)
def find_sharp_edges(points: np.ndarray, normals: np.ndarray, neighbors: np.ndarray, angle_threshold_rad: float) -> np.ndarray:
    n_points = len(points)
    is_edge = np.zeros(n_points, dtype=numba.boolean)

    for i in numba.prange(n_points):
        ni = normals[i]

        for j_idx in range(1, neighbors.shape[1]):
            j = neighbors[i, j_idx]
            nj = normals[j]

            dot_product = np.abs(np.dot(ni, nj))

            angle = np.arccos(dot_product)

            if angle > angle_threshold_rad:
                is_edge[i] = True
                break

    return is_edge

def fit_plane_to_points(points: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=3)
    pca.fit(points)

    pca_normal = pca.components_[-1]
    centroid = pca.mean_

    d = -np.dot(pca_normal, centroid)
    return np.append(pca_normal, d)


def merge_similar_planes(planes: List[np.ndarray], normal_threshold: float = 0.99, d_threshold: float = 0.05) -> np.ndarray:
    merged_planes = []
    processed_indices = np.zeros(len(planes), dtype=bool)

    for i in range(len(planes)):
        if processed_indices[i]:
            continue

        current_plane = planes[i]
        similar_planes_group = [current_plane]
        processed_indices[i] = True

        for j in range(i + 1, len(planes)):
            if processed_indices[j]:
                continue

            other_plane = planes[j]

            dot_product = np.abs(np.dot(current_plane[:3], other_plane[:3]))

            dist_diff = np.abs(current_plane[3] - other_plane[3])

            if dot_product > normal_threshold and dist_diff < d_threshold:
                similar_planes_group.append(other_plane)
                processed_indices[j] = True

        merged_plane = np.mean(similar_planes_group, axis=0)
        merged_planes.append(merged_plane)

    return np.array(merged_planes)

def voronoi_from_points(
    points: torch.Tensor,
    k_for_normals: int = 5,
    k_for_edges: int = 10,
    angle_threshold_degrees: float = 35.0,
    visualize: bool = False,
    verbose: bool = False
) -> List[torch.Tensor]:
    points_np = points.cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=k_for_normals))
    pcd.orient_normals_consistent_tangent_plane(k=k_for_normals)
    normals = np.asarray(pcd.normals)

    kdtree = cKDTree(points_np)
    _, neighbors = kdtree.query(points_np, k=k_for_edges)
    angle_threshold_rad = np.deg2rad(angle_threshold_degrees)
    is_edge_mask = find_sharp_edges(points_np, normals, neighbors, angle_threshold_rad)
    edge_points = points_np[is_edge_mask]

    edge_pcd = o3d.geometry.PointCloud()
    edge_pcd.points = o3d.utility.Vector3dVector(edge_points)
    edge_kdtree = cKDTree(edge_points)

    if visualize:
        o3d.visualization.draw_geometries([edge_pcd])

    distances, _ = edge_kdtree.query(edge_points, k=5)
    adjecency_radius = distances.max(axis=1).mean()

    condensed_distances = pdist(edge_points, 'euclidean')
    distance_matrix = squareform(condensed_distances)

    adjecency_matrix = np.where(distance_matrix < adjecency_radius, True, False).astype(np.bool)

    components = find_connected_components(edge_points, adjecency_matrix)

    fitted_planes = []
    for c in components:
        plane_eq = fit_plane_to_points(c)
        fitted_planes.append(plane_eq)

    fitted_planes = merge_similar_planes(fitted_planes)

    if len(fitted_planes) == 0:
        return [points]

    plane_normals = fitted_planes[:, :3]
    plane_offsets = fitted_planes[:, 3]

    signed_distances = np.einsum('ij,kj->ik', points_np, plane_normals) + plane_offsets
    side_of_plane_matrix = (signed_distances > 0)
    cell_labels = side_of_plane_matrix.dot(1 << np.arange(side_of_plane_matrix.shape[-1] - 1, -1, -1))

    unique_cell_labels = np.unique(cell_labels)

    voronoi_cells = []
    for i, label in enumerate(unique_cell_labels):
        cell_mask = (cell_labels == label)
        cell_points = points_np[cell_mask]

        cell_kdtree = cKDTree(cell_points)

        distances, _ = cell_kdtree.query(cell_points, k=5)
        adjecency_radius = distances.max(axis=1).mean()

        condensed_distances = pdist(cell_points, 'euclidean')
        distance_matrix = squareform(condensed_distances)

        adjecency_matrix = np.where(distance_matrix < adjecency_radius, True, False).astype(np.bool)

        components = find_connected_components(cell_points, adjecency_matrix)

        for c in components:
            voronoi_cells.append(torch.tensor(c, dtype=torch.float32, device="cuda"))

    if verbose:
        print(f"Extracted {len(voronoi_cells)} voronoi cells")

    return voronoi_cells

# TODO cleanup

def _calculate_canonical_plane(
    points: np.ndarray, tol: float = 1e-9
) -> Optional[Tuple[np.ndarray, float]]:
    if points.shape[0] < 3:
        return None, None

    for i in range(points.shape[0] - 2):
        p1, p2, p3 = points[i], points[i + 1], points[i + 2]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)

        if norm > tol:
            normal /= norm
            d = -np.dot(normal, p1)

            first_nonzero_idx = -1
            for j in range(normal.shape[0]):
                if not np.isclose(normal[j], 0, atol=tol):
                    first_nonzero_idx = j
                    break

            if first_nonzero_idx != -1 and normal[first_nonzero_idx] < 0:
                normal *= -1
                d *= -1

            return normal, d

    return None, None


def group_coplanar_points(
    points_sets: np.ndarray, tol: float = 1e-4
) -> Tuple[List[np.ndarray], List[List[int]]]:
    if not isinstance(points_sets, np.ndarray) or points_sets.ndim != 3:
        raise ValueError("Input must be a 3D NumPy array of shape (N, M, 3).")

    unique_planes = []

    for idx, point_set in enumerate(points_sets):
        current_normal, current_d = _calculate_canonical_plane(point_set)
        if current_normal is None:
            continue

        found_match = False
        for i in range(len(unique_planes)):
            existing_normal, existing_d = unique_planes[i][0]
            if np.allclose(current_normal, existing_normal, atol=tol) and np.isclose(
                current_d, existing_d, atol=tol
            ):
                unique_planes[i][1].append(point_set)
                unique_planes[i][2].append(idx)
                found_match = True
                break

        if not found_match:
            unique_planes.append(((current_normal, current_d), [point_set], [idx]))

    grouped_points = [np.vstack(points_list) for _, points_list, _ in unique_planes]
    grouped_indices = [indices_list for _, _, indices_list in unique_planes]

    return grouped_points, grouped_indices


def fit_cylinder(
    points: np.ndarray, mse_threshold: float = 1e-5, inlier_threshold: float = 1e-3
):
    if points.shape[1] != 3:
        raise ValueError("Input array must have shape (N, 3)")
    if len(points) < 7:
        raise ValueError("At least 7 points are required to fit a cylinder.")

    center_init = np.mean(points, axis=0)
    centered_points = points - center_init
    _, _, vh = np.linalg.svd(centered_points)

    dir1 = vh[0]
    dists1 = np.linalg.norm(
        centered_points - np.dot(centered_points, dir1[:, np.newaxis]) * dir1, axis=1
    )
    var1 = np.var(dists1)

    dir2 = vh[2]
    dists2 = np.linalg.norm(
        centered_points - np.dot(centered_points, dir2[:, np.newaxis]) * dir2, axis=1
    )
    var2 = np.var(dists2)

    if var1 < var2:
        direction_init = dir1
        radius_init = np.mean(dists1)
    else:
        direction_init = dir2
        radius_init = np.mean(dists2)

    def residuals(params, data_points):
        cx, cy, cz, dx, dy, dz, r = params
        center = np.array([cx, cy, cz])
        direction = np.array([dx, dy, dz])
        direction_norm = np.linalg.norm(direction)
        if direction_norm == 0:
            return np.full(data_points.shape[0], 1e6)
        direction = direction / direction_norm
        vec_p_c = data_points - center
        parallel_component = np.dot(vec_p_c, direction)
        dist_to_axis = np.linalg.norm(
            vec_p_c - parallel_component[:, np.newaxis] * direction, axis=1
        )
        return dist_to_axis - r

    x0 = np.concatenate([center_init, direction_init, [radius_init]])
    res = least_squares(residuals, x0, args=(points,))
    optimized_params = res.x
    center_fit = optimized_params[:3]
    direction_fit = optimized_params[3:6]
    radius_fit = optimized_params[6]

    if radius_fit < 0:
        radius_fit = -radius_fit

    direction_fit /= np.linalg.norm(direction_fit)
    final_residuals = residuals(optimized_params, points)
    mse = np.mean(final_residuals**2)
    point_errors = np.abs(final_residuals)
    inlier_mask = point_errors < inlier_threshold
    inliers = points[inlier_mask]
    is_cylinder = mse < mse_threshold

    return is_cylinder, radius_fit, center_fit, direction_fit, inliers


def components(edge_points, k: int = 5):
    edge_kdtree = cKDTree(edge_points)
    distances, _ = edge_kdtree.query(edge_points, k=k)
    adjecency_radius = distances.max(axis=1).mean()
    condensed_distances = pdist(edge_points, "euclidean")
    distance_matrix = squareform(condensed_distances)
    adjecency_matrix = np.where(distance_matrix < adjecency_radius, True, False).astype(
        np.bool
    )
    components = find_connected_components(edge_points, adjecency_matrix)
    return components


def are_coplanar(points: np.ndarray, tol: float = 1e-4) -> bool | np.ndarray:
    if not isinstance(points, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    is_single_set = False
    if points.ndim == 2:
        if points.shape[1] != 3:
            raise ValueError("For a single set, input array must have shape (M, 3).")
        is_single_set = True
        points = points[np.newaxis, :, :]

    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("Input array must have shape (M, 3) or (N, M, 3).")

    N, M, _ = points.shape

    if M <= 3:
        result = np.ones(N, dtype=bool)
        return result[0] if is_single_set else result

    p0s = points[:, 0:1, :]
    vector_matrices = points[:, 1:, :] - p0s
    ranks = np.linalg.matrix_rank(vector_matrices, tol=tol)
    result = ranks <= 2

    return result[0] if is_single_set else result

def spherical_to_cartesian(spherical_coords: np.ndarray) -> np.ndarray:
    r = spherical_coords[:, 0]
    theta = spherical_coords[:, 1]
    phi = spherical_coords[:, 2]

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    cartesian_coords = np.stack((x, y, z), axis=1)

    return cartesian_coords


def cartesian_to_spherical(cartesian_coords: np.ndarray) -> np.ndarray:
    x = cartesian_coords[:, 0]
    y = cartesian_coords[:, 1]
    z = cartesian_coords[:, 2]

    r = np.sqrt(x**2 + y**2 + z**2)

    r_safe = np.where(r == 0, 1e-15, r)
    theta = np.arccos(z / r_safe)
    phi = np.arctan2(y, x)

    spherical_coords = np.stack((r, theta, phi), axis=1)

    return spherical_coords


def cartesian_to_cylindrical(points, origin, direction):
    points = np.asarray(points)
    origin = np.asarray(origin)
    direction = np.asarray(direction)

    axis_vec = direction / np.linalg.norm(direction)

    vecs_from_origin = points - origin

    z_coords = np.dot(vecs_from_origin, axis_vec)

    projections = z_coords[:, np.newaxis] * axis_vec

    perp_vecs = vecs_from_origin - projections

    r_coords = np.linalg.norm(perp_vecs, axis=1)

    ref_vec_seed = np.array([1.0, 0.0, 0.0])
    if np.allclose(np.abs(np.dot(axis_vec, ref_vec_seed)), 1.0):
        ref_vec_seed = np.array([0.0, 1.0, 0.0])

    u_ref = np.cross(axis_vec, ref_vec_seed)
    u_ref /= np.linalg.norm(u_ref)

    v_ref = np.cross(axis_vec, u_ref)

    x_cyl = np.dot(perp_vecs, u_ref)
    y_cyl = np.dot(perp_vecs, v_ref)

    theta_coords = np.arctan2(y_cyl, x_cyl)

    return np.stack((r_coords, theta_coords, z_coords), axis=1)

def cylindrical_to_cartesian(points, origin, direction):
    points = np.asarray(points)
    origin = np.asarray(origin)
    direction = np.asarray(direction)

    axis_vec = direction / np.linalg.norm(direction)

    r_coords = points[:, 0]
    theta_coords = points[:, 1]
    z_coords = points[:, 2]

    ref_vec_seed = np.array([1.0, 0.0, 0.0])
    if np.allclose(np.abs(np.dot(axis_vec, ref_vec_seed)), 1.0):
        ref_vec_seed = np.array([0.0, 1.0, 0.0])

    u_ref = np.cross(axis_vec, ref_vec_seed)
    u_ref /= np.linalg.norm(u_ref)

    v_ref = np.cross(axis_vec, u_ref)

    x_cyl = r_coords * np.cos(theta_coords)
    y_cyl = r_coords * np.sin(theta_coords)

    projections = z_coords[:, np.newaxis] * axis_vec
    perp_vecs = x_cyl[:, np.newaxis] * u_ref + y_cyl[:, np.newaxis] * v_ref

    return origin + projections + perp_vecs


def voronoi_from_points_2(points: torch.Tensor):
    point_np = points.cpu().numpy()
    kdtree = cKDTree(point_np)
    distances, indices = kdtree.query(point_np, k=25)
    mask = are_coplanar(point_np[indices])
    coplanar_sets = point_np[indices[mask]]
    coplanar_groups, coplanar_indices = group_coplanar_points(coplanar_sets)
    coplanar_points = [torch.from_numpy(np.unique(g, axis=0)) for g in coplanar_groups]
    all_coplanar_points = np.unique(np.array([p for g in coplanar_groups for p in g]), axis=0)
    aset = set([tuple(x) for x in point_np])
    bset = set([tuple(x) for x in all_coplanar_points])
    non_coplanar_points = np.array([x for x in aset - bset])
    cs = components(torch.from_numpy(non_coplanar_points).float())
    cylinders = []
    non_cylinders = []

    for c in cs:
        is_cylinder, radius_fit, center_fit, direction_fit, inliers = fit_cylinder(c.cpu().numpy())
        if is_cylinder:
            cylinders.append(c)
        else:
            p = c.cpu().numpy()
            cyl = cartesian_to_cylindrical(p, center_fit, direction_fit)

            cyl[:, 0] = cyl[:, 0] * 1000

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cyl)

            clusters = pcd.cluster_dbscan(
                eps=5 * np.median(pcd.compute_nearest_neighbor_distance()),
                min_points=50,
            )

            for idx in np.unique(clusters):
                cluster_points = p[clusters == idx]

                if idx == -1:
                    non_cylinders += components(torch.from_numpy(cluster_points).float())
                else:
                    non_cylinders.append(torch.from_numpy(cluster_points))

    return non_cylinders + cylinders + coplanar_points


if __name__ == "__main__":
    pass