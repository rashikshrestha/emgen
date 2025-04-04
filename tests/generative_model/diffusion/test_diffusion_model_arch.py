
from PIL import Image
import torch
from torchvision import transforms

from emgen.generative_model.diffusion.diffusion_model_arch import *


def test_unetarch():
    model = UNetArch(
        device = 'cpu',
        in_channels=1, base_channels=64, 
        emb_size=128, num_down=2,
        weights = None
    )
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    #! Data Preparation
    img0 = Image.open('/home/rashik_shrestha/ws/emgen/test_data/mnist_4.png')
    img1 = Image.open('/home/rashik_shrestha/ws/emgen/test_data/mnist_5.png')
    img0_tensor = transform(img0)
    img1_tensor = transform(img1)
    batch = torch.stack([img0_tensor, img1_tensor], dim=0)
    assert batch.shape == torch.Size([2, 1, 28, 28])
    t = torch.tensor([12, 27], dtype=torch.float32)
    assert t.shape == torch.Size([2])
    
    #! Model Test
    out = model(batch, t)
    assert out.shape == batch.shape