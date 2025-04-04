from dataclasses import dataclass

@dataclass
class ToyDatasetConfig:
    _target_: str = "emgen.dataset.toy_dataset.ToyDataset"
    name: str = "dino"
    num: int = 8000
    
@dataclass
class MNISTDatasetConfig:
    _target_: str = "emgen.dataset.mnist.MNISTDataset"
    train: bool = True