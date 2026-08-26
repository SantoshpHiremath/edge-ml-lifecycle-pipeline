"""A small, real PyTorch CNN and its train/evaluate loop."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import CLASSES


class TinyEdgeCNN(nn.Module):
    """Deliberately small: two conv blocks + one FC layer, sized for a
    16x16 grayscale input so the exported model stays edge-appropriate
    (small enough to plausibly run on a Raspberry-Pi-class device)."""

    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 4 * 4, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv2(x)))   # 8x8 -> 4x4
        x = x.flatten(1)
        return self.fc(x)


def train_model(train_images, train_labels, epochs=8, lr=1e-2, seed=0):
    torch.manual_seed(seed)
    model = TinyEdgeCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x = torch.from_numpy(train_images)
    y = torch.from_numpy(train_labels)

    losses = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    return model, losses


def evaluate_model(model, images, labels):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(images))
        preds = logits.argmax(dim=1).numpy()
    accuracy = float((preds == labels).mean())
    return {"accuracy": accuracy, "n": len(labels)}
