import os
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class MNISTDataset(Dataset):
    def __init__(self, train=True):
        transform = transforms.ToTensor()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.mnist = datasets.MNIST(
            root=os.path.join(current_dir, "data"),
            train=train, 
            transform=transform, 
            download=True
        )
        

    def __len__(self):
        return len(self.mnist)


    def __getitem__(self, idx):
        image, _ = self.mnist[idx]
        return image