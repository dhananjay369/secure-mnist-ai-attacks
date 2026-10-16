import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from src.model import MNISTCNN
from src.utils import device, evaluate_model

dev = device()

poisoned = torch.load(
    "datasets/poisoned_mnist.pt",
    map_location="cpu",
    weights_only=True
)

x_train = poisoned["data"].float() / 255.0
y_train = poisoned["targets"].long()

if x_train.ndim == 3:
    x_train = x_train.unsqueeze(1)

train_dataset = TensorDataset(x_train, y_train)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

model = MNISTCNN().to(dev)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

for epoch in range(3):
    model.train()

    for x, y in train_loader:
        x = x.to(dev)
        y = y.to(dev)

        optimizer.zero_grad()

        output = model(x)
        loss = criterion(output, y)

        loss.backward()
        optimizer.step()

    print(f"poisoned training epoch {epoch + 1}")

torch.save(
    model.state_dict(),
    "models/poisoned_model.pt"
)

test_dataset = datasets.MNIST(
    "datasets",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False
)

metrics = evaluate_model(
    model,
    test_loader,
    dev
)

metrics["attack"] = "label poisoning"
metrics["poisoned_samples"] = 313
metrics["source_label"] = 7
metrics["target_label"] = 1

print(metrics)