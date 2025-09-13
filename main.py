import open3d as o3d
from pathlib import Path
from nac import TrainingConfig, preprocess_mesh, VoronoiNetwork, DataSampler, voronoi_from_points, reconstruct_mesh, show

if __name__ == "__main__":
    input_path = Path("/home/mikolaj/Documents/github/inr_voronoi/data/sphylinder/sphylinder.obj")

    config = TrainingConfig()
    config.epochs = 250

    mesh = o3d.io.read_triangle_mesh(input_path)
    preprocess_mesh(mesh)

    point_cloud = mesh.sample_points_poisson_disk(number_of_points=20000)
    point_cloud.estimate_normals()
    point_cloud.orient_normals_consistent_tangent_plane(5)

    pcs = voronoi_from_points(point_cloud)
    networks = [VoronoiNetwork() for _ in pcs]

    for nn, pc in zip(networks, pcs):
        data_sampler = DataSampler(pc, config)
        nn.to(config.device)
        nn.train_point_cloud(config, data_sampler)

    reconstruction = reconstruct_mesh(networks)
    o3d.io.write_triangle_mesh("/home/mikolaj/Downloads/rec.obj", reconstruction)
