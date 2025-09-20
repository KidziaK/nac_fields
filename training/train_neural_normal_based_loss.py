import open3d as o3d
from pathlib import Path
from nac import TrainingConfig, preprocess_mesh, VoronoiNetwork, DataSampler, voronoi_from_points, estimate_normals
from nac.metrics import chamfer_distance, ChamferDistanceMethod
from nac.reconstruction import reconstruct_mesh

if __name__ == "__main__":
    input_path = Path("/home/mikolaj/Documents/github/inr_voronoi/data/sphylinder/sphylinder.obj")

    config = TrainingConfig()
    config.epochs = 10000

    mesh = o3d.io.read_triangle_mesh(input_path)
    preprocess_mesh(mesh)

    point_cloud = mesh.sample_points_poisson_disk(number_of_points=20000)
    estimate_normals(point_cloud)

    pcs = voronoi_from_points(point_cloud)
    networks = [VoronoiNetwork() for _ in pcs]

    for nn, pc in zip(networks, pcs):
        data_sampler = DataSampler(pc, config)
        nn.to(config.device)
        nn.train_point_cloud(config, data_sampler)

    reconstruction = reconstruct_mesh(networks)

    print(f"chamfer distance l1: {chamfer_distance(mesh, reconstruction)}")
    print(f"chamfer distance l2: {chamfer_distance(mesh, reconstruction, method=ChamferDistanceMethod.L2)}")

    o3d.io.write_triangle_mesh("/home/mikolaj/Downloads/reconstruction.obj", reconstruction)
