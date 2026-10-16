
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.model import MNISTCNN, load_weights
from src.utils import device, save_json

dev = device()

test_ds = datasets.MNIST(
    "datasets",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

loader = DataLoader(
    test_ds,
    batch_size=256,
    shuffle=False
)

models = {}

for name, path in [
    ("Baseline", "models/baseline_model.pt"),
    ("Poisoned", "models/poisoned_model.pt")
]:
    model = MNISTCNN().to(dev)
    load_weights(model, path, dev)
    model.eval()
    models[name] = model

results = {}

for name, model in models.items():
    total = 0
    correct = 0
    actual_seven = 0
    seven_correct = 0
    seven_predicted_one = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(dev)
            y = y.to(dev)

            predictions = model(x).argmax(dim=1)

            correct += (predictions == y).sum().item()
            total += y.size(0)

            mask = y == 7
            actual_seven += mask.sum().item()
            seven_correct += (
                (predictions[mask] == 7).sum().item()
            )
            seven_predicted_one += (
                (predictions[mask] == 1).sum().item()
            )

    results[name] = {
        "clean_accuracy": correct / total,
        "class_7_accuracy": seven_correct / actual_seven,
        "actual_7_predicted_1": seven_predicted_one,
        "actual_seven_samples": actual_seven
    }

comparison = {
    "baseline": results["Baseline"],
    "poisoned": results["Poisoned"],
    "clean_accuracy_change_percentage_points": (
        results["Poisoned"]["clean_accuracy"]
        - results["Baseline"]["clean_accuracy"]
    ) * 100,
    "class_7_accuracy_change_percentage_points": (
        results["Poisoned"]["class_7_accuracy"]
        - results["Baseline"]["class_7_accuracy"]
    ) * 100
}

save_json(comparison, "results/poisoning_evaluation.json")

print("\nTargeted Data Poisoning Evaluation")
print("----------------------------------")

for name, metrics in results.items():
    print(f"\n{name} model")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

print("\nChange in clean accuracy (percentage points):")
print(comparison["clean_accuracy_change_percentage_points"])

print("\nChange in class 7 accuracy (percentage points):")
print(comparison["class_7_accuracy_change_percentage_points"])

print("\nSaved: results/poisoning_evaluation.json")