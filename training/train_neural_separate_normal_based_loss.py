import open3d as o3d
from pathlib import Path
from nac import TrainingConfig, preprocess_mesh, VoronoiNetwork, DataSampler, estimate_normals
from nac.reconstruction import reconstruct_mesh, prune

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
        center, scale_factor = preprocess_mesh(mesh)

        point_cloud = mesh.sample_points_poisson_disk(number_of_points=10000)
        estimate_normals(point_cloud)

        nn = VoronoiNetwork()

        data_sampler = DataSampler(point_cloud, config)
        nn.to(config.device)
        nn.train_point_cloud(config, data_sampler)

        reconstruction = reconstruct_mesh(nn)
        pruned_reconstruction = prune(reconstruction, point_cloud)

        pruned_reconstruction.scale(1 / scale_factor, center=[0, 0, 0])
        pruned_reconstruction.translate(center)

        o3d.io.write_triangle_mesh(f"reconstruction_{input_path.stem}.obj", pruned_reconstruction)
