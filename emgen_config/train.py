from dataclasses import dataclass

@dataclass
class TrainConfig:
    train_batch_size: int = 32
    eval_batch_size: int = 1000
    num_epochs: int = 200
    learning_rate: float = 1e-3
    save_images_step: int = 1