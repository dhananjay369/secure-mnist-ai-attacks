from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from src.model import MNISTCNN, load_weights, save_weights
from src.utils import device, evaluate_model, save_json, set_seed
from attacks.fgsm_attack import fgsm


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"


def main(epochs=3, epsilon=0.20, max_adv=10000):

    set_seed(42)
    dev = device()

    train_ds = datasets.MNIST(
        DATA,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    test_ds = datasets.MNIST(
        DATA,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    baseline = MNISTCNN().to(dev)
    load_weights(baseline, MODELS / "baseline_model.pt", dev)
    baseline.eval()

    model = MNISTCNN().to(dev)
    load_weights(model, MODELS / "baseline_model.pt", dev)

    adv_x = []
    adv_y = []
    seen = 0

    for x, y in DataLoader(
        train_ds,
        batch_size=256,
        shuffle=False
    ):

        if seen >= max_adv:
            break

        count = min(len(y), max_adv - seen)

        x = x[:count].to(dev)
        y = y[:count].to(dev)

        x_adv = fgsm(
            baseline,
            x,
            y,
            epsilon
        )

        adv_x.append(x_adv.cpu())
        adv_y.append(y.cpu())

        seen += count

    adv_x = torch.cat(adv_x)
    adv_y = torch.cat(adv_y)

    clean_x = train_ds.data[:seen]
    clean_x = clean_x.unsqueeze(1).float() / 255.0

    clean_y = train_ds.targets[:seen]

    combined_x = torch.cat(
        [clean_x, adv_x]
    )

    combined_y = torch.cat(
        [clean_y, adv_y]
    )

    train_loader = DataLoader(
        TensorDataset(combined_x, combined_y),
        batch_size=256,
        shuffle=True
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        model.train()

        for x, y in train_loader:

            x = x.to(dev)
            y = y.to(dev)

            optimizer.zero_grad()

            output = model(x)
            loss = criterion(output, y)

            loss.backward()
            optimizer.step()

        print("defense epoch", epoch + 1)

    save_weights(
        model,
        MODELS / "defended_model.pt"
    )

    clean_metrics = evaluate_model(
        model,
        DataLoader(
            test_ds,
            batch_size=256,
            shuffle=False
        ),
        dev
    )

    clean_metrics.update({
        "defense": "adversarial training",
        "epsilon": epsilon,
        "adv_training_samples": seen
    })

    save_json(
        clean_metrics,
        RESULTS / "defended_clean_metrics.json"
    )

    print("Clean test results:")
    print(clean_metrics)


if __name__ == "__main__":
    main()