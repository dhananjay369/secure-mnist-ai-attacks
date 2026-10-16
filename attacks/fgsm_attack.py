from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from src.model import MNISTCNN,load_weights
from src.utils import device,save_json,set_seed
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'datasets'; MODELS=ROOT/'models'; RESULTS=ROOT/'results'
def fgsm(model,x,y,epsilon):
    x=x.detach().clone().requires_grad_(True); loss=nn.CrossEntropyLoss()(model(x),y); model.zero_grad(set_to_none=True); loss.backward(); return (x+epsilon*x.grad.sign()).clamp(0,1).detach()
def main(epsilon=.20,max_samples=5000):
    set_seed(42); dev=device(); ds=datasets.MNIST(DATA,train=False,download=True,transform=transforms.ToTensor()); loader=DataLoader(ds,batch_size=256,shuffle=False); m=load_weights(MNISTCNN().to(dev),MODELS/'baseline_model.pt',dev); m.eval(); cc=ac=changed=total=0
    for x,y in loader:
        if total>=max_samples: break
        x,y=x[:max_samples-total].to(dev),y[:max_samples-total].to(dev)
        with torch.no_grad(): cp=m(x).argmax(1)
        xa=fgsm(m,x,y,epsilon)
        with torch.no_grad(): ap=m(xa).argmax(1)
        cc+=(cp==y).sum().item(); ac+=(ap==y).sum().item(); changed+=(cp!=ap).sum().item(); total+=len(y)
    r={'attack':'FGSM','epsilon':epsilon,'samples':total,'clean_accuracy':cc/total,'adversarial_accuracy':ac/total,'prediction_change_rate':changed/total,'attack_success_rate_on_clean_correct':(cc-ac)/cc if cc else 0}; RESULTS.mkdir(exist_ok=True); save_json(r,RESULTS/'fgsm_metrics.json'); print(r)
if __name__=='__main__': main()
