"""Read-only downstream audit of the latest prepared recording split."""
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import csv,json,hashlib
import cv2
import numpy as np
BASE=Path(__file__).resolve().parent


def main():
    candidates=[p for p in BASE.glob('day5_prepared_*') if (p/'preparation_report.json').is_file()]
    if not candidates: raise FileNotFoundError('No completed prepared split')
    dataset=max(candidates,key=lambda p:p.name)
    all_rows=[]; per_split={}; records={}; hashes=defaultdict(set)
    for split in ('train','validation','test'):
        with (dataset/f'{split}.csv').open() as f: rows=list(csv.DictReader(f))
        assert rows and all(r['split']==split for r in rows)
        assert len({(r['recording_id'],r['frame_index']) for r in rows})==len(rows)
        assert len({r['image_path'] for r in rows})==len(rows)
        records[split]={r['recording_id'] for r in rows}
        for r in rows:
            p=(dataset/r['image_path']).resolve()
            assert p.is_relative_to(dataset.resolve()) and p.is_file()
            payload=p.read_bytes(); assert hashlib.sha256(payload).hexdigest()==r['image_sha256']
            im=cv2.imdecode(np.frombuffer(payload,dtype=np.uint8),cv2.IMREAD_COLOR)
            assert im is not None and im.shape==(540,960,3)
            h=hashlib.sha256(im.tobytes()).hexdigest(); assert h==r['decoded_sha256']
            hashes[h].add(split)
            force=np.array([float(r[k]) for k in ('fx','fy','fz')]); assert np.isfinite(force).all()
            dt=(int(r['force_record_time_ns'])-int(r['image_record_time_ns']))/1e6
            assert abs(dt-float(r['signed_dt_ms']))<1e-6
            assert abs(abs(dt)-float(r['absolute_dt_ms']))<1e-6
            assert r['inside_force_time_coverage']=='True'
        all_rows.extend(rows)
        per_split[split]={'images':len(rows),'recordings':sorted(records[split])}
        # Data-loader smoke check: first image -> RGB float CHW, three force targets.
        first=cv2.imread(str(dataset/rows[0]['image_path']))
        sample=cv2.cvtColor(cv2.resize(first,(224,224)),cv2.COLOR_BGR2RGB).astype(np.float32)/255
        sample=np.transpose(sample,(2,0,1))
        assert sample.shape==(3,224,224) and 0<=sample.min()<=sample.max()<=1
        print('Audited',split,len(rows),'images; loader sample',sample.shape,flush=True)
    assert records['train'].isdisjoint(records['validation'])
    assert records['train'].isdisjoint(records['test'])
    assert records['validation'].isdisjoint(records['test'])
    assert all(len(s)==1 for s in hashes.values())
    with (dataset/'manifest.csv').open() as f: combined=list(csv.DictReader(f))
    assert sorted((r['image_path'],r['split']) for r in combined)==sorted((r['image_path'],r['split']) for r in all_rows)
    assert len(list((dataset/'images').rglob('*.jpg')))==len(all_rows)
    params=json.loads((dataset/'train_normalization.json').read_text())
    train=np.array([[float(r[k]) for k in ('fx','fy','fz')] for r in all_rows if r['split']=='train'])
    mean=np.array(params['mean']);std=np.array(params['std'])
    assert np.allclose(mean,train.mean(axis=0)) and np.allclose(std,train.std(axis=0)) and np.all(std>0)
    assert np.allclose(((train-mean)/std)*std+mean,train,atol=1e-10)
    report={'status':'PASS','dataset':str(dataset),'split_counts':per_split,'total_images':len(all_rows),
        'all_images_readable_and_hash_verified':True,'cross_split_recording_overlap':0,'cross_split_exact_decoded_image_overlap':0,
        'all_pairing_gaps_recomputed':True,'normalization_fit_verified':'train only','loader_shape':[3,224,224],
        'scope':'Integrity and loader checks only; no training, performance evaluation, or acquisition-latency validation.'}
    out=BASE/('day5_split_audit_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(exist_ok=False)
    (out/'audit_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));print('AUDIT_DIR='+str(out))


if __name__=='__main__':main()
