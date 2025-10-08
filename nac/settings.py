from pydantic import BaseModel


class TrainingConfig(BaseModel):
    device: str = "cuda"
    epochs: int = 10000
    learning_rate: float = 5e-5
    offset: float = 0.0


class ReconstructionConfig(BaseModel):
    device: str = "cuda"
    padding: float = 0.1
    batch_size: int = 1
    grid_resolution: int = 128
    samples: int = 10000
    level_set: float = 0.0

    @property
    def bounding_box_extent(self) -> float:
        return 1.0 + self.padding
