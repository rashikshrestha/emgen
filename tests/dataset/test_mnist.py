import torch
from emgen.dataset.mnist import MNISTDataset
from torchvision.utils import save_image


def test_mnist():
    dataset = MNISTDataset()
    assert len(dataset)==60000
    assert dataset[0].shape == torch.Size([1, 28, 28])
    assert dataset[0].dtype == torch.float32
    assert dataset[0].min() == 0 
    assert dataset[0].max() == 1