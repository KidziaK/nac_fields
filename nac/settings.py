from pydantic import BaseModel


class TrainingConfig(BaseModel):
    device: str = "cuda"
    epochs: int = 10000
    learning_rate: float = 5e-5
    gradient_clip: float = 10.0
    non_manifold_alpha: float = 100.0


class ReconstructionConfig(BaseModel):
    device: str = "cuda"
    padding: float = 0.05
    grid_resolution: int = 256
    samples: int = 10000
    offset: float = 0.0

    @property
    def bounding_box_extent(self) -> float:
        return 1.0 + self.padding + self.offset
