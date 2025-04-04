import torch
from emgen.dataset.mnist import MNISTDataset
from torchvision.utils import save_image


def test_mnist():
    dataset = MNISTDataset()
    assert len(dataset)==60000
    data = dataset[0]
    assert data.shape == torch.Size([1, 28, 28])
    assert data.dtype == torch.float32
    assert data.min() == 0 
    assert data.max() == 1