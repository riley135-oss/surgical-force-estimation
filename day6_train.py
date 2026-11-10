"""CPU pilot: train-mean baseline versus small CNN; validation only, test untouched."""
import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader


class Frames(Dataset):
    def __init__(self, root, split, mean, std):
        self.root = root
        with (root / f'{split}.csv').open() as f:
            self.rows = list(csv.DictReader(f))
        self.y = np.array([[float(r[k]) for k in ('fx', 'fy', 'fz')] for r in self.rows], dtype=np.float32)
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        with Image.open(self.root / self.rows[i]['image_path']) as image:
            x = np.asarray(image.convert('RGB').resize((128, 128)), dtype=np.float32) / 255.0
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy((self.y[i] - self.mean) / self.std)


def metrics(pred, truth):
    error = pred - truth
    return {'mae_per_axis': np.abs(error).mean(0).tolist(),
            'rmse_per_axis': np.sqrt((error ** 2).mean(0)).tolist(),
            'macro_mae': float(np.abs(error).mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--seed', type=int, default=6001)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(True)
    out = Path(__file__).resolve().parent / ('day6_run_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir()
    with (a.data / 'train.csv').open() as f:
        rows = list(csv.DictReader(f))
    y = np.array([[float(r[k]) for k in ('fx', 'fy', 'fz')] for r in rows], dtype=np.float32)
    mean, std = y.mean(0), np.maximum(y.std(0), 1e-6)
    train, val = Frames(a.data, 'train', mean, std), Frames(a.data, 'validation', mean, std)
    assert set(r['recording_id'] for r in train.rows).isdisjoint(r['recording_id'] for r in val.rows)
    baseline = metrics(np.broadcast_to(mean, val.y.shape), val.y)
    model = nn.Sequential(nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
                          nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                          nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                          nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
                          nn.Linear(1024, 64), nn.ReLU(), nn.Linear(64, 3))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader = DataLoader(train, batch_size=32, shuffle=True, num_workers=2,
                              generator=torch.Generator().manual_seed(a.seed))
    val_loader = DataLoader(val, batch_size=64, shuffle=False, num_workers=2)
    best, history = float('inf'), []
    print('Device: CPU | train:', len(train), '| validation:', len(val), flush=True)
    print('Constant baseline:', baseline, flush=True)
    for epoch in range(1, a.epochs + 1):
        model.train()
        total = 0.0
        for x, target in train_loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x), target)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
        model.eval()
        with torch.no_grad():
            predictions = np.concatenate([model(x).numpy() for x, _ in val_loader]) * std + mean
        score = metrics(predictions, val.y)
        history.append({'epoch': epoch, 'train_normalized_mse': total / len(train), **score})
        print(f'Epoch {epoch}/{a.epochs}: validation MAE={score["macro_mae"]:.6f}', flush=True)
        if score['macro_mae'] < best:
            best = score['macro_mae']
            best_score, best_epoch = score, epoch
            np.save(out / 'validation_predictions.npy', predictions)
            torch.save({'state_dict': model.state_dict(), 'mean': mean.tolist(), 'std': std.tolist(),
                        'epoch': epoch, 'image_size': 128}, out / 'best_model.pt')
        (out / 'history.json').write_text(json.dumps(history, indent=2))
    result = {'seed': a.seed, 'torch_version': torch.__version__, 'device': 'cpu',
              'data': str(a.data.resolve()), 'train_count': len(train), 'validation_count': len(val),
              'baseline': baseline, 'cnn_validation': best_score, 'best_epoch': best_epoch,
              'selection': 'minimum validation macro MAE', 'test_evaluated': False,
              'limits': 'Single seed; four-recording pilot; bag-time pairing, latency and force axes unvalidated.'}
    (out / 'metrics.json').write_text(json.dumps(result, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r['epoch'] for r in history], [r['macro_mae'] for r in history], marker='o', label='Small CNN')
    ax.axhline(baseline['macro_mae'], linestyle='--', color='gray', label='Train-mean baseline')
    ax.set(xlabel='Epoch', ylabel='Validation macro MAE (nominal N)')
    ax.legend(); fig.tight_layout(); fig.savefig(out / 'validation_mae.png', dpi=160); plt.close(fig)
    conclusion = (f'本次 Day 6 在学校 CPU 上比较训练集均值基线与小型 CNN，使用 {len(train)} 张训练图像、'
                  f'{len(val)} 张验证图像。均值基线验证集三轴平均 MAE 为 {baseline["macro_mae"]:.4f} N，'
                  f'CNN 在第 {best_epoch} 个 epoch 达到最低验证 MAE {best:.4f} N。'
                  + ('本次验证结果支持将 CNN 作为后续候选模型。' if best < baseline['macro_mae'] else '本次 CNN 未优于均值基线，暂保留均值基线。')
                  + '该结果仅来自单随机种子和少量录像，且最佳 epoch 由同一验证集选择；'
                  '不能视为独立测试性能。图像与力仅按 bag 记录时间配对，传感器延迟及坐标轴尚未验证。'
                  '测试集未读取或评估，下一步应固定实验方案并进行多随机种子验证，再进行一次最终测试。')
    (out / 'CONCLUSION.md').write_text(conclusion + '\n')
    print(conclusion, flush=True)
    print('Results saved to:', out, flush=True)


if __name__ == '__main__':
    main()
