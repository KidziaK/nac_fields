from pathlib import Path
import open3d as o3d
import torch
from nac import TrainingConfig, preprocess_mesh, estimate_normals, FlatCAD
from nac.data import SirenDataset
from nac.settings import ReconstructionConfig
from nac.reconstruction import reconstruct_mesh

if __name__ == "__main__":
    part_name = "00000003"
    input_path = Path(__file__).parents[1] / "data" / part_name / f"{part_name}_ground_truth.obj"
    output_name = f"flatcad_{part_name}.obj"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    offset = 0.0
    training_config = TrainingConfig(epochs=1000, device=device, offset=offset)
    reconstruction_config = ReconstructionConfig(device=device, level_set=-offset)

    mesh = o3d.io.read_triangle_mesh(input_path)
    mesh = preprocess_mesh(mesh)

    point_cloud = mesh.sample_points_poisson_disk(number_of_points=30000)
    pcd = estimate_normals(point_cloud)

    dataset = SirenDataset(config=reconstruction_config, point_cloud=point_cloud)

    network = FlatCAD()
    network.train_point_cloud(config=training_config, dataset=dataset)

    reconstruction = reconstruct_mesh(network, config=reconstruction_config)
    o3d.io.write_triangle_mesh(output_name, reconstruction)
