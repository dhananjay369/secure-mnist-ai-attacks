from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.model import MNISTCNN, save_weights
from src.utils import device, save_json, set_seed
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'datasets'; MODELS=ROOT/'models'; RESULTS=ROOT/'results'
def main():
    set_seed(42); dev=device(); ds=datasets.MNIST(DATA,train=True,download=True,transform=transforms.ToTensor()); loader=DataLoader(ds,batch_size=128,shuffle=True)
    model=MNISTCNN().to(dev); opt=torch.optim.Adam(model.parameters(),lr=1e-3); crit=nn.CrossEntropyLoss(); hist=[]
    for epoch in range(1,4):
        model.train(); ls=correct=total=0
        for x,y in loader:
            x,y=x.to(dev),y.to(dev); opt.zero_grad(); out=model(x); loss=crit(out,y); loss.backward(); opt.step(); ls+=loss.item()*y.size(0); correct+=(out.argmax(1)==y).sum().item(); total+=y.size(0)
        r={'epoch':epoch,'loss':ls/total,'accuracy':correct/total}; hist.append(r); print(r)
    MODELS.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True); save_weights(model,MODELS/'baseline_model.pt'); save_json(hist,RESULTS/'training_history.json')
if __name__=='__main__': main()
