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


def train(model, loader, epochs=2, T_=1.0, soft_targets=None, dev='cpu'):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        model.train()
        for i, (x, y) in enumerate(loader):
            x, y = x.to(dev), y.to(dev)
            logits = model(x) / T_
            if soft_targets is None:
                loss = F.cross_entropy(logits, y)
            else:
                # distillation loss: KL between student and teacher soft logits
                target = soft_targets(x) / T_
                loss = F.kl_div(F.log_softmax(logits, -1),
                                F.softmax(target, -1), reduction='batchmean')
            opt.zero_grad()
            loss.backward()
            opt.step()


def fgsm(model, x, y, eps, T_=1.0):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x) / T_, y)
    g = torch.autograd.grad(loss, x)[0]
    return (x + eps * g.sign()).clamp(0, 1).detach()


def cw_logit_attack(model, x, y, eps=0.3, alpha=0.01, steps=40):
    """Direct logit-difference attack — bypasses high-T gradient masking."""
    x_orig = x.clone().detach()
    x = x_orig + torch.empty_like(x).uniform_(-eps, eps)
    x = x.clamp(0, 1)
    for _ in range(steps):
        x = x.detach().requires_grad_(True)
        logits = model(x)  # no /T scaling — operate on raw logits
        true_logit = logits.gather(1, y.unsqueeze(1)).squeeze(1)
        other_max = logits.scatter(1, y.unsqueeze(1), -1e9).max(1)[0]
        loss = (true_logit - other_max).mean()  # minimize -> attack
        g = torch.autograd.grad(loss, x)[0]
        x = (x - alpha * g.sign())
        x = (x_orig + (x - x_orig).clamp(-eps, eps)).clamp(0, 1)
    return x.detach()


def evaluate(model, attack_fn, loader, dev):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        x_adv = attack_fn(model, x, y)
        with torch.no_grad():
            correct += (model(x_adv).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader = loaders()

    print('training teacher (T=%.0f)...' % T)
    teacher = CNN().to(dev)
    train(teacher, train_loader, epochs=2, T_=T, dev=dev)

    print('distilling student...')
    student = CNN().to(dev)
    train(student, train_loader, epochs=2, T_=T, dev=dev,
          soft_targets=lambda x: teacher(x).detach())

    print('FGSM @ eps=0.3 (T-scaled): acc =',
          evaluate(student, lambda m, x, y: fgsm(m, x, y, 0.3, T_=T),
                   test_loader, dev))
    print('CW-style logit attack:    acc =',
          evaluate(student, lambda m, x, y: cw_logit_attack(m, x, y),
                   test_loader, dev))


if __name__ == '__main__':
    main()
