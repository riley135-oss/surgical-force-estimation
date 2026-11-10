"""One frozen test evaluation; no training or checkpoint selection."""
import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from day6_transfer import Images


def metrics(p,y):
    e=p-y
    return {'macro_mae':float(np.abs(e).mean()),'pooled_rmse':float(np.sqrt((e**2).mean())),
            'axis_mae':np.abs(e).mean(0).tolist(),'axis_rmse':np.sqrt((e**2).mean(0)).tolist()}


def main():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser()
    parser.add_argument('--heads',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    args=parser.parse_args()
    heads=args.heads.resolve()
    out=root/('day6_test_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')); out.mkdir()
    protocol=(root/'TEST_PROTOCOL.md').read_bytes()
    (out/'TEST_PROTOCOL.md').write_bytes(protocol)
    checkpoint_hashes={str(s):hashlib.sha256((heads/f'head_{s}.pt').read_bytes()).hexdigest() for s in (6201,6202,6203)}
    (out/'frozen_hashes.json').write_text(json.dumps({'protocol':hashlib.sha256(protocol).hexdigest(),'heads':checkpoint_hashes},indent=2))
    rows={}
    for split in ('train','test'):
        with (args.data/f'{split}.csv').open() as f: rows[split]=list(csv.DictReader(f))
    assert len(rows['test'])==1733
    assert set(r['recording_id'] for r in rows['train']).isdisjoint(r['recording_id'] for r in rows['test'])
    for r in rows['test']:
        assert hashlib.sha256((args.data/r['image_path']).read_bytes()).hexdigest()==r['image_sha256']
    print('Verified test image hashes:',len(rows['test']),flush=True)
    y_train=np.array([[float(r[k]) for k in ('fx','fy','fz')] for r in rows['train']])
    y=np.array([[float(r[k]) for k in ('fx','fy','fz')] for r in rows['test']])
    baselines={}
    for label,c in [('mean',y_train.mean(0)),('median',np.median(y_train,axis=0))]:
        baselines[label]=metrics(np.broadcast_to(c,y.shape),y)
        baselines[label]['constant']=c.tolist()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    weights=ResNet18_Weights.IMAGENET1K_V1
    model=resnet18(weights=weights); model.fc=nn.Identity(); model.eval()
    loader=DataLoader(Images(args.data,rows['test'],weights.transforms()),batch_size=32,num_workers=2)
    parts=[]
    with torch.inference_mode():
        for i,xb in enumerate(loader):
            parts.append(model(xb).numpy())
            if (i+1)%20==0: print('Test features:',min((i+1)*32,len(y)),'/',len(y),flush=True)
    features=np.concatenate(parts).astype(np.float64)
    scores=[]
    predictions={}
    for seed in (6201,6202,6203):
        checkpoint=torch.load(heads/f'head_{seed}.pt',map_location='cpu',weights_only=True)
        head=nn.Linear(512,3); head.load_state_dict(checkpoint['state_dict']); head.eval()
        x=torch.tensor((features-np.array(checkpoint['feature_mean']))/np.array(checkpoint['feature_std']),dtype=torch.float32)
        with torch.inference_mode(): pred=head(x).numpy()
        assert np.isfinite(pred).all()
        predictions[seed]=pred
        np.save(out/f'test_predictions_{seed}.npy',pred)
        scores.append({'seed':seed,'epoch':checkpoint['epoch'],**metrics(pred,y)})
    mean=float(np.mean([s['macro_mae'] for s in scores])); sd=float(np.std([s['macro_mae'] for s in scores],ddof=1))
    report={'test_recordings':sorted(set(r['recording_id'] for r in rows['test'])),'test_count':len(y),
            'baselines':baselines,'seeds':scores,'mean_macro_mae':mean,'sample_sd_macro_mae':sd,
            'mean_pooled_rmse':float(np.mean([s['pooled_rmse'] for s in scores])),
            'sample_sd_pooled_rmse':float(np.std([s['pooled_rmse'] for s in scores],ddof=1)),
            'selection_on_test':False,'image_hashes_verified':len(y)}
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=np.array([int(r['image_record_time_ns']) for r in rows['test']]); t=(t-t[0])/1e9
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
    for j,ax in enumerate(axes):
        ax.plot(t,y[:,j],color='black',label='Recorded force',linewidth=1)
        ax.axhline(baselines['median']['constant'][j],color='gray',linestyle='--',label='Train median')
        for seed,pred in predictions.items(): ax.plot(t,pred[:,j],alpha=.65,linewidth=.8,label=f'L1 seed {seed}')
        ax.set_ylabel(('Fx','Fy','Fz')[j]+' (nominal N)')
    axes[0].legend(ncol=5,fontsize=8); axes[-1].set_xlabel('Held-out test recording time (s)')
    fig.tight_layout(); fig.savefig(out/'test_predictions.png',dpi=160); plt.close(fig)
    improvement=(1-mean/baselines['median']['macro_mae'])*100
    conclusion=(f'Day 6 固定模型后在 M2_L_V1_1 的 {len(y)} 张保留测试图像上评估。训练均值与中位数基线的测试 MAE '
                f'分别为 {baselines["mean"]["macro_mae"]:.4f}、{baselines["median"]["macro_mae"]:.4f} N；'
                f'冻结 ResNet18 加 MAE 线性头的三个种子平均测试 MAE 为 {mean:.4f} ± {sd:.4f} N，'
                f'相对中位数基线改善 {improvement:.1f}%。'
                + ('该候选在本次独立录像上降低了平均误差，但仍须检查峰值预测。' if improvement>0 else '该候选在本次独立录像上未超过中位数基线，验证集改善未转化为测试优势。')
                + '全部种子完整报告，没有根据测试结果重新选择模型。测试仅含一个录像，种子间标准差不代表跨组织泛化置信区间；'
                '时间配对、传感器延迟和力坐标轴限制仍存在。当前测试录像已使用，后续调参应另设新的独立测试录像。')
    (out/'CONCLUSION.md').write_text(conclusion+'\n')
    print(json.dumps(report,indent=2),flush=True); print(conclusion,flush=True); print('Results saved to:',out,flush=True)


if __name__=='__main__': main()
