import open3d as o3d
import torch
import numpy as np
from pathlib import Path
from nac.settings import TrainingConfig
from nac.network import  FlatCAD
from nac.preprocess import preprocess_mesh
from nac.normals import estimate_normals
from nac.data import SirenDataset
from nac.settings import ReconstructionConfig
from nac.reconstruction import reconstruct_mesh
from nac.metrics import chamfer_distance, ChamferDistanceMethod

if __name__ == "__main__":
    part_name = "sphylinder"
    input_path = Path(__file__).parents[1] / "data" / part_name / f"{part_name}_ground_truth.obj"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for offset in np.linspace(0.0, 0.05, num=4):
        training_config = TrainingConfig(epochs=5000, device=device)
        reconstruction_config = ReconstructionConfig(device=device, offset=offset)

        mesh = o3d.io.read_triangle_mesh(input_path)
        mesh = preprocess_mesh(mesh)

        point_cloud = mesh.sample_points_poisson_disk(number_of_points=30000)
        pcd = estimate_normals(point_cloud)

        dataset = SirenDataset(config=reconstruction_config, point_cloud=point_cloud)

        network = FlatCAD()
        network.train_point_cloud(config=training_config, dataset=dataset)

        reconstruction = reconstruct_mesh(network, config=reconstruction_config)
        reconstruction = preprocess_mesh(reconstruction)
        c1 = chamfer_distance(mesh, reconstruction, method=ChamferDistanceMethod.L1)
        c2 = chamfer_distance(mesh, reconstruction, method=ChamferDistanceMethod.L2)

        print(c1, c2)

        output_name = f"with_original_{part_name}_{offset}_{c1}_{c2}.obj"
        o3d.io.write_triangle_mesh(output_name, reconstruction)

