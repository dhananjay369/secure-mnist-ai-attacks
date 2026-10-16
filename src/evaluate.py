from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from src.model import MNISTCNN,load_weights
from src.utils import device,evaluate_model,save_confusion_matrix,save_json
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'datasets'; MODELS=ROOT/'models'; RESULTS=ROOT/'results'
def main():
    dev=device(); ds=datasets.MNIST(DATA,train=False,download=True,transform=transforms.ToTensor()); loader=DataLoader(ds,batch_size=256,shuffle=False)
    m=load_weights(MNISTCNN().to(dev),MODELS/'baseline_model.pt',dev); r=evaluate_model(m,loader,dev); RESULTS.mkdir(exist_ok=True); save_json(r,RESULTS/'baseline_metrics.json'); save_confusion_matrix(r['confusion_matrix'],RESULTS/'baseline_confusion_matrix.png','Baseline MNIST Confusion Matrix'); print(r)
if __name__=='__main__': main()
