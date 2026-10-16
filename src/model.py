import torch
from torch import nn

class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64*7*7, 128), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128, 10)
        )
    def forward(self, x): return self.classifier(self.features(x))

def save_weights(model, path): torch.save(model.state_dict(), path)
def load_weights(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True)); return model
