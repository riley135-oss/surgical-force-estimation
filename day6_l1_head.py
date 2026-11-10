"""Compare MAE-trained linear heads on frozen cached image features. No test data."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def main():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser()
    parser.add_argument('--features',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    args=parser.parse_args()
    feature_dir=args.features.resolve()
    out=root/('day6_l1_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')); out.mkdir()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    x=np.load(feature_dir/'train_features.npy')
    v=np.load(feature_dir/'validation_features.npy')
    # Reuse the exact train-only normalization from the ridge experiment.
    prep=np.load(feature_dir/'ridge_model.npz')
    x=torch.tensor((x-prep['feature_mean'])/prep['feature_std'],dtype=torch.float32)
    v=torch.tensor((v-prep['feature_mean'])/prep['feature_std'],dtype=torch.float32)
    ys={}
    for split in ('train','validation'):
        with (args.data/f'{split}.csv').open() as f:
            rows=list(csv.DictReader(f))
        ys[split]=torch.tensor([[float(r[k]) for k in ('fx','fy','fz')] for r in rows])
    assert len(x)==len(ys['train']) and len(v)==len(ys['validation'])
    assert torch.isfinite(x).all() and torch.isfinite(v).all()
    median=ys['train'].median(dim=0).values
    # numpy median handles even sample counts; train count here is odd.
    median_mae=float((ys['validation']-median).abs().mean())
    ridge=json.loads((feature_dir/'metrics.json').read_text())
    results=[]
    for seed in (6201,6202,6203):
        torch.manual_seed(seed)
        model=nn.Linear(x.shape[1],3)
        # Starting at the train median provides a meaningful no-image starting point.
        nn.init.zeros_(model.weight)
        with torch.no_grad(): model.bias.copy_(median)
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-2)
        loader=DataLoader(TensorDataset(x,ys['train']),batch_size=128,shuffle=True,
                          generator=torch.Generator().manual_seed(seed))
        best=median_mae
        best_epoch=0
        best_pred=median.expand_as(ys['validation']).clone()
        best_state={k:z.detach().clone() for k,z in model.state_dict().items()}
        history=[]
        for epoch in range(1,101):
            model.train(); loss_sum=0.
            for xb,yb in loader:
                optimizer.zero_grad()
                loss=(model(xb)-yb).abs().mean()
                loss.backward(); optimizer.step()
                loss_sum+=float(loss.detach())*len(xb)
            model.eval()
            with torch.inference_mode():
                pred=model(v)
                val_mae=float((pred-ys['validation']).abs().mean())
            history.append({'epoch':epoch,'train_mae':loss_sum/len(x),'validation_mae':val_mae})
            if val_mae<best:
                best=val_mae; best_epoch=epoch; best_pred=pred.clone()
                best_state={k:z.detach().clone() for k,z in model.state_dict().items()}
            if epoch%25==0:
                print(f'Seed {seed}, epoch {epoch}: validation={val_mae:.6f}, best={best:.6f}',flush=True)
        err=(best_pred-ys['validation']).numpy()
        result={'seed':seed,'best_epoch':best_epoch,'macro_mae':best,
                'axis_mae':np.abs(err).mean(0).tolist(),
                'pooled_rmse':float(np.sqrt((err**2).mean()))}
        results.append(result)
        torch.save({'state_dict':best_state,'feature_mean':prep['feature_mean'].tolist(),
                    'feature_std':prep['feature_std'].tolist(),'seed':seed,'epoch':best_epoch},out/f'head_{seed}.pt')
        np.save(out/f'validation_predictions_{seed}.npy',best_pred.numpy())
        (out/f'history_{seed}.json').write_text(json.dumps(history,indent=2))
        print('Completed:',result,flush=True)
    scores=[r['macro_mae'] for r in results]
    mean,sd=float(np.mean(scores)),float(np.std(scores,ddof=1))
    report={'method':'Frozen ResNet18 + linear MAE head','seeds':results,'mean_macro_mae':mean,
            'sample_sd_macro_mae':sd,'median_baseline_mae':median_mae,
            'ridge_validation_mae':ridge['selected']['macro_mae'],'epochs':100,
            'lr':1e-3,'weight_decay':1e-2,'batch_size':128,'initialization':'zero weights, train median bias',
            'selection':'minimum validation MAE including epoch0 constant median','test_evaluated':False}
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    for r in results:
        hist=json.loads((out/f'history_{r["seed"]}.json').read_text())
        ax.plot([0]+[h['epoch'] for h in hist],[median_mae]+[h['validation_mae'] for h in hist],label=str(r['seed']))
    ax.axhline(median_mae,color='black',linestyle='--',label='Train median')
    ax.set(xlabel='Epoch',ylabel='Validation macro MAE (nominal N)'); ax.legend()
    fig.tight_layout(); fig.savefig(out/'l1_learning_curves.png',dpi=160); plt.close(fig)
    conclusion=(f'固定 ResNet18 图像特征并将线性预测层改为 MAE 损失，三个随机种子各训练 100 轮，'
                f'最佳验证 MAE 平均为 {mean:.4f} ± {sd:.4f} N；中位数基线为 {median_mae:.4f} N，'
                f'岭回归为 {ridge["selected"]["macro_mae"]:.4f} N。'
                '所有种子均从训练集中位数初始化，并将第 0 轮常数预测纳入检查点选择；'
                '因此该选择机制下不劣于基线不能本身证明图像有效。'
                '检查点由同一验证录像选择，经过多次验证实验存在选择偏差；测试集未读取。'
                '下一步应固定候选方案，再用独立录像或保留测试集评估，不能继续依赖同一验证集宣称泛化改善。')
    (out/'CONCLUSION.md').write_text(conclusion+'\n')
    print(conclusion,flush=True); print('Results saved to:',out,flush=True)


if __name__=='__main__': main()
