from pathlib import Path
import torch
from torchvision import datasets,transforms
from src.utils import save_json,set_seed
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'datasets'; RESULTS=ROOT/'results'
def main(fraction=.05,source=7,target=1):
    set_seed(42); ds=datasets.MNIST(DATA,train=True,download=True,transform=transforms.ToTensor()); labels=ds.targets.clone(); candidates=torch.where(labels==source)[0]; n=max(1,int(len(candidates)*fraction)); labels[candidates[:n]]=target; DATA.mkdir(exist_ok=True); torch.save({'data':ds.data,'targets':labels},DATA/'poisoned_mnist.pt'); r={'source_label':source,'target_label':target,'poison_fraction_of_source_class':fraction,'poisoned_samples':n,'original_training_size':len(ds)}; save_json(r,RESULTS/'poisoning_metadata.json'); print(r)
if __name__=='__main__': main()
