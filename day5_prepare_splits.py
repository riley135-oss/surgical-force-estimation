"""Reproducible small recording-level split; no model training or label filtering.
Run from project root: python day5/day5_prepare_splits.py
Download is restricted to the four pinned public source files below.
"""
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import importlib.metadata
import json
import os
import shutil
import urllib.request
import time

BASE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(BASE.parent / '.mplconfig'))
import cv2
import numpy as np
from rosbags.highlevel import AnyReader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RECORDS = [
    ('M1_L_V1_1', 'train', '61d598d3c62631d16410a49194466d44'),
    ('M3_L_V1_1', 'train', '041b19b69ee7014cf5cc3903e0e85190'),
    ('M1_L_V1_2', 'validation', 'd6d873d66af15d3ab59e3503bccfa097'),
    ('M2_L_V1_1', 'test', 'de858f10e65cac815ba03878be4ad6e1'),
]
IMAGE_TOPIC = '/camera/left/image_color/compressed'
FORCE_TOPIC = '/force_sensor'
RAW = BASE / 'data_raw'
FIELDS = ['split','recording_id','material_id','arm','viewpoint','frame_index','image_path',
          'image_record_time_ns','force_record_time_ns','signed_dt_ms','absolute_dt_ms',
          'inside_force_time_coverage','fx','fy','fz','image_sha256','decoded_sha256',
          'image_header_time_ns','force_header_time_ns']


def checksum(path, algorithm='sha256'):
    digest = hashlib.new(algorithm)
    with path.open('rb') as file:
        for data in iter(lambda: file.read(1024*1024), b''):
            digest.update(data)
    return digest.hexdigest()


def download(item):
    name, split, expected = item
    target = RAW / f'{name}.bag'
    url = f'https://zenodo.org/records/21041361/files/{name}.bag?download=1'
    if target.exists():
        if checksum(target, 'md5') != expected:
            raise RuntimeError(f'Existing source checksum mismatch; unchanged: {target}')
        print(f'Verified cached source: {name}', flush=True)
    else:
        # Exclusive temporary file, retained on failure for diagnosis.
        temporary = RAW / f'{name}.{time.time_ns()}.part'
        request = urllib.request.Request(url, headers={'User-Agent': 'Day5DatasetPreparation/1.0'})
        print(f'Downloading {name}', flush=True)
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open('xb') as file:
            shutil.copyfileobj(response, file, length=1024*1024)
        if checksum(temporary, 'md5') != expected:
            raise RuntimeError(f'New download checksum mismatch: {temporary}')
        if target.exists():
            raise FileExistsError(target)
        temporary.rename(target)
        print(f'Downloaded and MD5 verified: {name}', flush=True)
    return {'recording_id': name, 'split': split, 'url': url, 'md5': expected,
            'sha256': checksum(target), 'bytes': target.stat().st_size}


def header_ns(message):
    header = getattr(message, 'header', None)
    if header is None:
        return None
    value = int(header.stamp.sec)*10**9 + int(header.stamp.nanosec)
    return value if value > 0 else None


def write_csv(path, fields, rows):
    with path.open('x', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def extract_record(item, out):
    name, split, _ = item
    bag = RAW / f'{name}.bag'
    forces, images = [], []
    output_images = out / 'images' / name
    output_images.mkdir(parents=True, exist_ok=False)
    shapes = Counter()
    with AnyReader([bag]) as reader:
        selected = [c for c in reader.connections if c.topic in (IMAGE_TOPIC, FORCE_TOPIC)]
        if {c.topic for c in selected} != {IMAGE_TOPIC, FORCE_TOPIC}:
            raise RuntimeError(f'Missing required topics: {name}')
        for connection, timestamp, raw in reader.messages(connections=selected):
            msg = reader.deserialize(raw, connection.msgtype)
            htime = header_ns(msg)
            if connection.topic == FORCE_TOPIC:
                f = msg.wrench.force
                forces.append((timestamp, htime, msg.header.frame_id, float(f.x), float(f.y), float(f.z)))
            else:
                payload = bytes(msg.data)
                if not payload.startswith(b'\xff\xd8\xff'):
                    raise RuntimeError('Expected original JPEG payload; no automatic re-encoding')
                image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f'Cannot decode {name} frame {len(images)}')
                if image.shape != (540,960,3):
                    raise RuntimeError(f'Unexpected dimensions: {image.shape}')
                shapes[str(image.shape)] += 1
                image_path = output_images / f'frame_{len(images):05d}.jpg'
                with image_path.open('xb') as file:
                    file.write(payload)
                # Read back saved bytes: verify filesystem export and decoding.
                if image_path.read_bytes() != payload or cv2.imread(str(image_path)) is None:
                    raise RuntimeError(f'Image export failed: {image_path}')
                images.append((timestamp, htime, msg.header.frame_id,
                    image_path.relative_to(out).as_posix(), hashlib.sha256(payload).hexdigest(),
                    hashlib.sha256(image.tobytes()).hexdigest()))
    if not forces or not images:
        raise RuntimeError(f'Empty stream: {name}')
    ft = np.array([f[0] for f in forces], dtype=np.int64)
    it = np.array([im[0] for im in images], dtype=np.int64)
    force_array = np.array([f[3:] for f in forces])
    assert np.all(np.diff(ft)>=0) and np.all(np.diff(it)>=0)
    assert np.isfinite(force_array).all()
    right = np.clip(np.searchsorted(ft,it,side='left'),0,len(ft)-1)
    left = np.clip(right-1,0,len(ft)-1)
    nearest = np.where(np.abs(it-ft[left])<=np.abs(ft[right]-it),left,right)
    coverage = (it>=ft[0]) & (it<=ft[-1])
    delta = (ft[nearest]-it)/1e6
    if not coverage.all():
        raise RuntimeError(f'Images outside force coverage: {name}; inspect before splitting')
    rows = []
    for idx,(im, fi) in enumerate(zip(images,nearest)):
        force = forces[int(fi)]
        rows.append(dict(zip(FIELDS,[split,name,name.split('_')[0],'L','V1',idx,im[3],int(im[0]),int(force[0]),
            float(delta[idx]),float(abs(delta[idx])),True,*force[3:],im[4],im[5],im[1],force[1]])))
    report = {'recording_id':name,'split':split,'image_count':len(images),'force_count':len(forces),
        'shapes':dict(shapes),'image_header_valid':sum(im[1] is not None for im in images),
        'force_header_valid':sum(f[1] is not None for f in forces),
        'image_frame_ids':dict(Counter(im[2] for im in images)),
        'force_frame_ids':dict(Counter(f[2] for f in forces)),
        'absolute_dt_median_ms':float(np.median(abs(delta))),
        'absolute_dt_p95_ms':float(np.percentile(abs(delta),95)),
        'absolute_dt_max_ms':float(np.max(abs(delta))),
        'outside_force_coverage':int(np.sum(~coverage)),
        'gap_over_5ms_count':int(np.sum(abs(delta)>5))}
    print(f'Exported and audited {name}: {len(images)} images',flush=True)
    return rows,report


def main():
    RAW.mkdir(exist_ok=True)
    if shutil.disk_usage(BASE).free < 4*1024**3:
        raise RuntimeError('Need at least 4 GiB free space')
    # Split is fixed before inspecting validation/test label distributions.
    run = BASE / ('day5_prepared_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run.mkdir(exist_ok=False)
    sources = list(ThreadPoolExecutor(max_workers=3).map(download,RECORDS))
    rows,reports = [],[]
    for item in RECORDS:
        new_rows,report = extract_record(item,run)
        rows.extend(new_rows); reports.append(report)
    record_sets = {split:{r['recording_id'] for r in rows if r['split']==split} for split in ('train','validation','test')}
    assert record_sets['train'].isdisjoint(record_sets['validation'])
    assert record_sets['train'].isdisjoint(record_sets['test'])
    assert record_sets['validation'].isdisjoint(record_sets['test'])
    image_groups=defaultdict(set)
    for r in rows:
        image_groups[r['decoded_sha256']].add(r['split'])
    duplicate_across_splits = sum(len(s)>1 for s in image_groups.values())
    if duplicate_across_splits:
        raise RuntimeError(f'{duplicate_across_splits} exact decoded images cross splits; review required')
    for split in record_sets:
        selected=[r for r in rows if r['split']==split]
        write_csv(run/f'{split}.csv',FIELDS,selected)
    write_csv(run/'manifest.csv',FIELDS,rows)
    # Fit normalization exclusively on train; validation/test remain untouched.
    train=np.array([[r['fx'],r['fy'],r['fz']] for r in rows if r['split']=='train'])
    normalization={'targets':['fx','fy','fz'],'fitted_on':'train only','mean':train.mean(axis=0).tolist(),
        'std':train.std(axis=0).tolist(),'application':'For training, z=(force-mean)/std; invert before N-based metrics. No normalized labels written.'}
    (run/'train_normalization.json').write_text(json.dumps(normalization,indent=2))
    report={'status':'PASS','records':reports,'sources':sources,'split_recordings':{k:sorted(v) for k,v in record_sets.items()},
        'split_image_counts':dict(Counter(r['split'] for r in rows)), 'total_images':len(rows),
        'decoded_image_hash_overlap_between_splits':duplicate_across_splits,
        'pairing':'nearest bag recording timestamp; ties select earlier force; offline only',
        'test_design':'Small unseen-material M2 test at same L arm/V1 view; not official full benchmark split',
        'image_export':'Original JPEG payload copied without re-encoding; original PNG pilot export preserved',
        'no_filtering_zeroing_or_coordinate_transform':True,
        'units':'N according to associated paper; raw sensor components, physical axis mapping unverified',
        'limitations':['Invalid header timestamps are tracked, not repaired. Recording-time gaps are not acquisition-latency validation.',
        'Exact hash non-overlap does not prove independence of scenes; validation shares M1 tissue/view with one train recording.',
        'Only four recordings; not a robust generalization benchmark. M2 was assigned test before label-distribution analysis.',
        'Sensor zero, mechanical-axis mapping and acquisition latency remain unverified; never interpret fz as normal force without calibration.'],
        'software':{k:importlib.metadata.version(k) for k in ('numpy','opencv-python','rosbags')},
        'pipeline_sha256':checksum(Path(__file__))}
    (run/'preparation_report.json').write_text(json.dumps(report,indent=2))
    # Descriptive split sizes only: no test-label exploration or model selection.
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    counts=report['split_image_counts']
    bars=ax.bar(list(counts),list(counts.values()),color=['steelblue','orange','seagreen'])
    ax.bar_label(bars); ax.set(ylabel='Image count',title='Recording-level pilot split | 2 train / 1 validation / 1 test')
    fig.savefig(run/'split_overview.png',dpi=160);plt.close(fig)
    print(json.dumps({'status':'PASS','counts':counts,'total_images':len(rows),'cross_split_hash_overlap':0},indent=2),flush=True)
    print('RESULT_DIR='+str(run),flush=True)


if __name__=='__main__':main()
