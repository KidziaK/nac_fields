import open3d as o3d
from pathlib import Path
from nac import TrainingConfig, preprocess_mesh, DataSampler, estimate_normals, FlatCAD
from nac.metrics import chamfer_distance, ChamferDistanceMethod
from nac.reconstruction import reconstruct_mesh

if __name__ == "__main__":
    input_path = Path("/home/mikolaj/Documents/github/inr_voronoi/data/sphylinder/sphylinder.obj")

    config = TrainingConfig()
    config.epochs = 100

    mesh = o3d.io.read_triangle_mesh(input_path)
    preprocess_mesh(mesh)

    point_cloud = mesh.sample_points_poisson_disk(number_of_points=50000)
    estimate_normals(point_cloud)

    data_sampler = DataSampler(point_cloud, config)
    nn = FlatCAD()
    nn.to(config.device)
    nn.train_point_cloud(config, data_sampler)

    reconstruction = reconstruct_mesh(nn)

    print(f"chamfer distance l1: {chamfer_distance(mesh, reconstruction)}")
    print(f"chamfer distance l2: {chamfer_distance(mesh, reconstruction, method=ChamferDistanceMethod.L2)}")

    o3d.io.write_triangle_mesh("/home/mikolaj/Downloads/reconstruction_flatcad.obj", reconstruction)
