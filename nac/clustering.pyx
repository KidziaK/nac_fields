import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt

ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def find_planar_clusters(np.ndarray[DTYPE_t, ndim=2] points,
                         double distance_threshold=0.05,
                         int min_points_in_cluster=50,
                         int max_iterations=1000):
    cdef int n_points = points.shape[0]
    if n_points < min_points_in_cluster:
        return [], points

    cdef DTYPE_t[:, :] points_view = points
    cdef np.ndarray[np.uint8_t, ndim=1] unclassified_mask = np.ones(n_points, dtype=np.uint8)

    cdef list clusters = []
    cdef int i, j
    cdef int p1_idx, p2_idx, p3_idx
    cdef DTYPE_t p1[3], p2[3], p3[3]
    cdef DTYPE_t v1[3], v2[3]
    cdef DTYPE_t normal[3]
    cdef DTYPE_t norm_len, d

    cdef np.ndarray[np.int32_t, ndim=1] current_inliers_indices
    cdef np.ndarray[np.int32_t, ndim=1] best_inliers_indices
    cdef np.ndarray[np.int32_t, ndim=1] unclassified_indices
    cdef int n_unclassified

    while True:
        unclassified_indices = np.where(unclassified_mask)[0].astype(np.int32)
        n_unclassified = unclassified_indices.shape[0]

        if n_unclassified < min_points_in_cluster:
            break

        best_inliers_indices = np.array([], dtype=np.int32)

        for i in range(max_iterations):
            if unclassified_indices.shape[0] < 3:
                break

            sampled_indices = np.random.choice(unclassified_indices, 3, replace=False)
            p1_idx = sampled_indices[0]
            p2_idx = sampled_indices[1]
            p3_idx = sampled_indices[2]

            for j in range(3):
                p1[j] = points_view[p1_idx, j]
                p2[j] = points_view[p2_idx, j]
                p3[j] = points_view[p3_idx, j]

            for j in range(3):
                v1[j] = p2[j] - p1[j]
                v2[j] = p3[j] - p1[j]

            normal[0] = v1[1] * v2[2] - v1[2] * v2[1]
            normal[1] = v1[2] * v2[0] - v1[0] * v2[2]
            normal[2] = v1[0] * v2[1] - v1[1] * v2[0]

            norm_len = sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
            if norm_len < 1e-9:
                continue

            for j in range(3):
                normal[j] /= norm_len

            d = -(normal[0] * p1[0] + normal[1] * p1[1] + normal[2] * p1[2])

            distances = np.abs(np.dot(points[unclassified_indices], normal) + d)
            inlier_mask = distances < distance_threshold
            current_inliers_indices = unclassified_indices[inlier_mask]

            if current_inliers_indices.shape[0] > best_inliers_indices.shape[0]:
                best_inliers_indices = current_inliers_indices

        if best_inliers_indices.shape[0] >= min_points_in_cluster:
            clusters.append(points[best_inliers_indices])
            unclassified_mask[best_inliers_indices] = 0
        else:
            break

    final_unclassified_indices = np.where(unclassified_mask)[0]
    unclassified_points = points[final_unclassified_indices]

    return clusters, unclassified_points
