"""Defensive distillation demo on MNIST.

Step 1: Train teacher with high-temperature softmax.
Step 2: Distill student from teacher's soft labels.
Step 3: FGSM appears to fail (gradients vanish under high T).
Step 4: Carlini-Wagner-style attack with logit-targeted loss bypasses it.

Lesson: gradient masking != robustness. This was understood by 2020 but the
worked example is still the cleanest way to grok why.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

T = 20.0  # distillation temperature


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


def loaders(bs=128):
    tfm = transforms.ToTensor()
    tr = datasets.MNIST('./data', train=True, download=True, transform=tfm)
    te = datasets.MNIST('./data', train=False, download=True, transform=tfm)
    return DataLoader(tr, batch_size=bs, shuffle=True), DataLoader(te, batch_size=bs)


