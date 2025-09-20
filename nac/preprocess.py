import open3d as o3d
import numpy as np


def preprocess_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.AxisAlignedBoundingBox:
    aabb = mesh.get_axis_aligned_bounding_box()
    center = aabb.get_center()
    mesh.translate(-center)

    aabb_centered = mesh.get_axis_aligned_bounding_box()
    extent = aabb_centered.get_extent()

    max_dimension = np.max(extent)
    scale_factor = 1.0 / max_dimension
    mesh.scale(scale_factor, center=np.array([0.0, 0.0, 0.0]))

    return aabb
