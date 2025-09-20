import open3d as o3d
from pathlib import Path
from nac import TrainingConfig, preprocess_mesh, VoronoiNetwork, DataSampler, estimate_normals
from nac.metrics import chamfer_distance, ChamferDistanceMethod
from nac.reconstruction import reconstruct_mesh

if __name__ == "__main__":
    gt_name = "sphylinder"
    data_path = Path(__file__).parents[1] / "data" / gt_name
    parts = ["disk", "cylinder", "hemisphere"]

    inputs = [
        data_path / f"{p}.obj" for p in parts
    ]

    config = TrainingConfig()
    config.epochs = 1000

    for input_path in inputs:
        mesh = o3d.io.read_triangle_mesh(input_path)
        aabb = preprocess_mesh(mesh)

        point_cloud = mesh.sample_points_poisson_disk(number_of_points=20000)
        estimate_normals(point_cloud)

        nn = VoronoiNetwork()

        data_sampler = DataSampler(point_cloud, config)
        nn.to(config.device)
        nn.train_point_cloud(config, data_sampler)

        reconstruction = reconstruct_mesh(nn, aabb=aabb)

        print(f"{input_path.stem} chamfer distance l1: {chamfer_distance(mesh, reconstruction)}")
        print(f"{input_path.stem} chamfer distance l2: {chamfer_distance(mesh, reconstruction, method=ChamferDistanceMethod.L2)}")

        o3d.io.write_triangle_mesh(f"/home/mikolaj/Downloads/reconstruction_{input_path.stem}.obj", reconstruction)
