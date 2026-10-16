
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.model import MNISTCNN, load_weights
from src.utils import device, validate_mnist_input, save_json
from attacks.fgsm_attack import fgsm

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

baseline = MNISTCNN().to(dev)
load_weights(baseline, "models/baseline_model.pt", dev)
baseline.eval()

defended = MNISTCNN().to(dev)
load_weights(defended, "models/defended_model.pt", dev)
defended.eval()

epsilon = 0.20

baseline_clean_correct = 0
baseline_adv_correct = 0
defended_clean_correct = 0
defended_adv_correct = 0
total = 0

for x, y in loader:
    x = x.to(dev)
    y = y.to(dev)

    validate_mnist_input(x)

    x_adv = fgsm(baseline, x, y, epsilon)

    with torch.no_grad():
        baseline_clean_pred = baseline(x).argmax(1)
        baseline_adv_pred = baseline(x_adv).argmax(1)
        defended_clean_pred = defended(x).argmax(1)
        defended_adv_pred = defended(x_adv).argmax(1)

    baseline_clean_correct += (
        baseline_clean_pred == y
    ).sum().item()

    baseline_adv_correct += (
        baseline_adv_pred == y
    ).sum().item()

    defended_clean_correct += (
        defended_clean_pred == y
    ).sum().item()

    defended_adv_correct += (
        defended_adv_pred == y
    ).sum().item()

    total += y.size(0)

results = {
    "epsilon": epsilon,
    "total_samples": total,
    "baseline_clean_accuracy": baseline_clean_correct / total,
    "baseline_fgsm_accuracy": baseline_adv_correct / total,
    "defended_clean_accuracy": defended_clean_correct / total,
    "defended_fgsm_accuracy": defended_adv_correct / total
}

results["fgsm_accuracy_improvement_percentage_points"] = (
    results["defended_fgsm_accuracy"]
    - results["baseline_fgsm_accuracy"]
) * 100

results["clean_accuracy_change_percentage_points"] = (
    results["defended_clean_accuracy"]
    - results["baseline_clean_accuracy"]
) * 100

save_json(results, "results/defense_comparison.json")

print("\nSame FGSM adversarial samples")
print("--------------------------------")
print("Baseline clean accuracy:",
      results["baseline_clean_accuracy"])
print("Baseline FGSM accuracy:",
      results["baseline_fgsm_accuracy"])
print("Defended clean accuracy:",
      results["defended_clean_accuracy"])
print("Defended FGSM accuracy:",
      results["defended_fgsm_accuracy"])
print("FGSM accuracy improvement (percentage points):",
      results["fgsm_accuracy_improvement_percentage_points"])
print("Clean accuracy change (percentage points):",
      results["clean_accuracy_change_percentage_points"])
print("\nSaved: results/defense_comparison.json")