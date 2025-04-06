import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_moons


class ToyDataset(Dataset):
    def __init__(self, name='dino', num=8000):
        if name == "moons":
            self.data = moons_dataset(num)
        elif name == "dino":
            self.data = dino_dataset(num)
        elif name == "line":
            self.data = line_dataset(num)
        elif name == "circle":
            self.data = circle_dataset(num)
        else:
            raise ValueError(f"Unknown dataset: {self.name}")

 
    def __len__(self):
        return self.data.shape[0]


    def __getitem__(self, idx):
        return self.data[idx]
    
        
def moons_dataset(n=8000):
    X, _ = make_moons(n_samples=n, random_state=42, noise=0.03)
    X[:, 0] = (X[:, 0] + 0.3) * 2 - 1
    X[:, 1] = (X[:, 1] + 0.3) * 3 - 1
    return torch.from_numpy(X.astype(np.float32))


def line_dataset(n=8000):
    rng = np.random.default_rng(42)
    x = rng.uniform(-0.5, 0.5, n)
    y = rng.uniform(-1, 1, n)
    X = np.stack((x, y), axis=1)
    X *= 4
    return torch.from_numpy(X.astype(np.float32))


def circle_dataset(n=8000):
    rng = np.random.default_rng(42)
    x = np.round(rng.uniform(-0.5, 0.5, n)/2, 1)*2
    y = np.round(rng.uniform(-0.5, 0.5, n)/2, 1)*2
    norm = np.sqrt(x**2 + y**2) + 1e-10
    x /= norm
    y /= norm
    theta = 2 * np.pi * rng.uniform(0, 1, n)
    r = rng.uniform(0, 0.03, n)
    x += r * np.cos(theta)
    y += r * np.sin(theta)
    X = np.stack((x, y), axis=1)
    X *= 3
    return torch.from_numpy(X.astype(np.float32))


def dino_dataset(n=8000):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "DatasaurusDozen.tsv")
    df = pd.read_csv(csv_path, sep="\t")
    df = df[df["dataset"] == "dino"]

    rng = np.random.default_rng(42)
    ix = rng.integers(0, len(df), n)
    x = df["x"].iloc[ix].tolist()
    x = np.array(x) + rng.normal(size=len(x)) * 0.15
    y = df["y"].iloc[ix].tolist()
    y = np.array(y) + rng.normal(size=len(x)) * 0.15
    x = (x/54 - 1) * 4
    y = (y/48 - 1) * 4
    X = np.stack((x, y), axis=1)
    return torch.from_numpy(X.astype(np.float32))


def get_gt_dino_dataset():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "DatasaurusDozen.tsv")
    df = pd.read_csv(csv_path, sep="\t")
    df = df[df["dataset"] == "dino"]

    x = np.array(df["x"].tolist())
    y = np.array(df["y"].tolist())
    x = (x/54 - 1) * 4
    y = (y/48 - 1) * 4
    return np.stack((x, y), axis=1)