"""Frozen ImageNet ResNet18 + ridge regression; train/validation only."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights


class Images(Dataset):
    def __init__(self, root, rows, transform):
        self.root, self.rows, self.transform = root, rows, transform
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        with Image.open(self.root / self.rows[i]['image_path']) as image:
            return self.transform(image.convert('RGB'))


def score(pred, y):
    err = pred-y
    return {'macro_mae': float(np.abs(err).mean()), 'axis_mae': np.abs(err).mean(0).tolist(),
            'axis_rmse': np.sqrt((err**2).mean(0)).tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(6101)
    torch.use_deterministic_algorithms(True)
    out = Path(__file__).resolve().parent / ('day6_transfer_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir()
    rows, y = {}, {}
    for split in ('train', 'validation'):
        with (a.data/f'{split}.csv').open() as f:
            rows[split] = list(csv.DictReader(f))
        y[split] = np.array([[float(r[k]) for k in ('fx','fy','fz')] for r in rows[split]])
    assert set(r['recording_id'] for r in rows['train']).isdisjoint(r['recording_id'] for r in rows['validation'])
    baseline = {}
    for label, constant in [('mean', y['train'].mean(0)), ('median', np.median(y['train'],axis=0))]:
        baseline[label] = score(np.broadcast_to(constant,y['validation'].shape),y['validation'])
        baseline[label]['constant'] = constant.tolist()
    print('Baselines:', baseline, flush=True)
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    features = {}
    for split in ('train','validation'):
        loader = DataLoader(Images(a.data, rows[split], weights.transforms()),batch_size=32,num_workers=2)
        parts=[]
        with torch.inference_mode():
            for i,x in enumerate(loader):
                parts.append(model(x).numpy())
                if (i+1)%20==0:
                    print(f'{split}: extracted {min((i+1)*32,len(rows[split]))}/{len(rows[split])}',flush=True)
        features[split]=np.concatenate(parts).astype(np.float64)
        np.save(out/f'{split}_features.npy',features[split])
    mean=features['train'].mean(0)
    std=np.maximum(features['train'].std(0),1e-6)
    x=(features['train']-mean)/std
    v=(features['validation']-mean)/std
    target_mean=y['train'].mean(0)
    gram=x.T@x
    rhs=x.T@(y['train']-target_mean)
    candidates=[]
    best=float('inf')
    # Fixed grid before observing validation results; no iterative expansion.
    for alpha in (1.0,10.0,100.0,1000.0,10000.0):
        coef=np.linalg.solve(gram+alpha*np.eye(x.shape[1]),rhs)
        pred=v@coef+target_mean
        metrics=score(pred,y['validation'])
        candidates.append({'alpha':alpha,**metrics})
        print('Ridge:', candidates[-1], flush=True)
        if metrics['macro_mae']<best:
            best=metrics['macro_mae']; chosen=candidates[-1]
            np.save(out/'validation_predictions.npy',pred)
            np.savez(out/'ridge_model.npz',coef=coef,feature_mean=mean,feature_std=std,target_mean=target_mean,alpha=alpha)
    report={'method':'Frozen ResNet18 ImageNet V1 + standardized-feature ridge',
            'preprocessing':'Official weights.transforms(): resize256, center crop224, ImageNet normalization',
            'seed':6101,'baselines':baseline,'candidates':candidates,'selected':chosen,
            'selection':'minimum validation macro MAE over fixed five-alpha grid',
            'train_count':len(rows['train']),'validation_count':len(rows['validation']),
            'test_evaluated':False,'weight_url':weights.url,'torch':torch.__version__}
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pred=np.load(out/'validation_predictions.npy')
    t=np.array([int(r['image_record_time_ns']) for r in rows['validation']]); t=(t-t[0])/1e9
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
    for j,ax in enumerate(axes):
        ax.plot(t,y['validation'][:,j],label='Recorded force',color='black',linewidth=1)
        ax.plot(t,pred[:,j],label='Frozen ResNet18 + ridge',linewidth=1)
        ax.axhline(baseline['median']['constant'][j],label='Train median',color='gray',linestyle='--')
        ax.set_ylabel(('Fx','Fy','Fz')[j]+' (nominal N)')
    axes[0].legend(ncol=3); axes[-1].set_xlabel('Validation recording time (s)')
    fig.tight_layout(); fig.savefig(out/'predictions.png',dpi=160); plt.close(fig)
    conclusion=(f'补充实验比较训练均值、训练中位数与冻结 ResNet18 特征加岭回归。验证集 MAE 分别为 '
                f'{baseline["mean"]["macro_mae"]:.4f}、{baseline["median"]["macro_mae"]:.4f}、{best:.4f} N；'
                f'岭回归在预先固定的五个正则化参数中选择 alpha={chosen["alpha"]:g}。'
                '所有拟合和标准化只使用训练集；预训练网络不进行微调，测试集未读取。'
                '验证结果同时参与参数选择，不能当作独立测试性能；仅有一个验证录像，'
                '仍需警惕背景或时序相关性，记录时间配对及力坐标轴限制保持不变。')
    (out/'CONCLUSION.md').write_text(conclusion+'\n')
    print(conclusion,flush=True); print('Results saved to:',out,flush=True)


if __name__=='__main__':
    main()
