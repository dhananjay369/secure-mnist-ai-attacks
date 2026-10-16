from pathlib import Path
import json, random, time
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def device(): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,'w',encoding='utf-8') as f: json.dump(data,f,indent=2)
def evaluate_model(model, loader, dev):
    model.eval(); criterion=torch.nn.CrossEntropyLoss(); loss_sum=correct=total=0; ys=[]; ps=[]
    start=time.perf_counter()
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(dev),y.to(dev); out=model(x); loss=criterion(out,y)
            loss_sum += loss.item()*y.size(0); pred=out.argmax(1); correct+=(pred==y).sum().item(); total+=y.size(0)
            ys += y.cpu().tolist(); ps += pred.cpu().tolist()
    elapsed=time.perf_counter()-start
    return {'loss':loss_sum/total,'accuracy':correct/total,'inference_time_seconds':elapsed,'milliseconds_per_image':elapsed/total*1000,'confusion_matrix':confusion_matrix(ys,ps).tolist()}
def save_confusion_matrix(cm,path,title):
    import matplotlib.pyplot as plt
    Path(path).parent.mkdir(parents=True,exist_ok=True); plt.figure(figsize=(7,6)); plt.imshow(cm); plt.title(title); plt.xlabel('Predicted'); plt.ylabel('True'); plt.colorbar()
    for i in range(10):
        for j in range(10): plt.text(j,i,str(cm[i][j]),ha='center',va='center')
    plt.tight_layout(); plt.savefig(path,dpi=160); plt.close()

def validate_mnist_input(x):
    if not torch.is_tensor(x):
        raise TypeError("Input must be a PyTorch tensor.")

    if x.ndim != 4 or x.shape[1:] != (1, 28, 28):
        raise ValueError("Input must have shape (N, 1, 28, 28).")

    if not torch.isfinite(x).all():
        raise ValueError("Input contains NaN or Inf values.")

    if x.min() < 0 or x.max() > 1:
        raise ValueError("Pixel values must be in the range [0, 1].")

    return True