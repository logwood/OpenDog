"""Warm-start a 128px landmark tracker from screened video pseudo labels.

Only the landmark FPN and heatmaps are updated by default. Backbone, presence,
visibility and box heads stay frozen; a frozen copy anchors unsupported points.
No input-size, inference graph or capture-gate change is needed. Checkpoint
selection uses video-disjoint validation only. Test videos are evaluated once
after selection, and never enter the training Dataset or sampler.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import json
import math
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dog_capture.engine.refine import apply, crop_matrix, warp_crop
from import_dog_landmark_weights import load_checkpoint, packed
from prepare_dog_video_distillation import sha256
from train_dog_face_landmarks import FLIP, SIZE, heat_targets


def load_rows(directory):
    root = Path(directory)
    metadata = json.loads((root/'dataset.json').read_text(encoding='utf8'))
    if sha256(root/'labels.jsonl') != metadata['labels_sha256']:
        raise ValueError('Dataset labels changed after preparation')
    rows = [json.loads(line) for line in (root/'labels.jsonl').read_text(encoding='utf8').splitlines() if line]
    sources = {r['id']:r for r in metadata['source_manifest']['sources']}
    protected = set(metadata['source_manifest'].get('protected_sha256', []))
    groups, images, videos = {}, set(), {}
    for row in rows:
        source = sources.get(row['video_id'])
        if source is None or any(row[key] != source[source_key] for key,source_key in
                                 [('split','split'),('dog_group','dog_group'),('source_sha256','sha256')]):
            raise ValueError(f'Label disagrees with fixed source manifest: {row["id"]}')
        if row['source_sha256'] in protected and row['split'] != 'test':
            raise ValueError('Protected video entered model development')
        if row['source_sha256'] in videos and videos[row['source_sha256']] != row['split']:
            raise ValueError('Source video leakage')
        videos[row['source_sha256']] = row['split']
        key = row['dog_group']
        if key in groups and groups[key] != row['split']:
            raise ValueError(f'Dog/video leakage: {key}')
        groups[key] = row['split']
        if row['image_sha256'] in images:
            raise ValueError(f'Duplicate image: {row["id"]}')
        images.add(row['image_sha256'])
        if sha256(root/row['image']) != row['image_sha256']:
            raise ValueError(f'Image hash mismatch: {row["id"]}')
    return rows, metadata


def yaw_numpy(points):
    axis = points[..., 1, :] - points[..., 0, :]
    delta = points[..., 2, :] - (points[..., 0, :] + points[..., 1, :]) / 2
    signed = (delta[..., 1]*axis[..., 0] - delta[..., 0]*axis[..., 1])
    direction = np.where(signed < 0, -1, 1)
    return (delta*axis).sum(-1) / np.maximum((axis*axis).sum(-1), 1e-8) * direction / 1.05


class VideoCrops:
    def __init__(self, root, rows, train=False, seed=20261113):
        self.root, self.rows, self.train, self.seed = Path(root), rows, train, seed

    def __len__(self):
        return len(self.rows)

    @lru_cache(maxsize=24)
    def image(self, name):
        frame = cv2.imread(str(self.root/name))
        if frame is None:
            raise ValueError(f'Cannot read training frame {name}')
        return frame

    def __getitem__(self, index):
        row = self.rows[index]
        rng = random if self.train else random.Random(self.seed + index)
        frame = self.image(row['image'])
        box = np.array(row['box'], np.float64)
        side = box[2]-box[0]
        centre = (box[:2]+box[2:])/2
        shift = .12 if self.train else .08
        centre += np.array([rng.uniform(-shift,shift),rng.uniform(-shift,shift)])*side
        side *= math.exp(rng.uniform(math.log(.8),math.log(1.25)))
        box = np.r_[centre-side/2,centre+side/2]
        # Preserve the original all-roll orientation task, but focus on crops
        # the deployed tracker sees after its upright recrop.
        residual = rng.uniform(-180,180) if self.train and rng.random()<.25 else rng.uniform(-25,25)
        angle = row['upright_angle'] + residual
        matrix = crop_matrix(box, angle, .35, SIZE)
        crop = warp_crop(frame, matrix, SIZE, antialias=True)
        points = apply(matrix,row['points']).astype(np.float32)/SIZE
        weights = np.array(row['point_weights'],np.float32)
        inside = ((points >= 0) & (points < 1)).all(1)
        weights *= inside
        weights *= abs(residual)<=55
        axis = np.stack([points[2],(points[0]+points[1])/2])
        axis_weights = np.array([row['point_weights'][2],min(row['point_weights'][:2])],np.float32)
        axis_weights *= ((axis>=0)&(axis<1)).all(1)
        if self.train:
            crop = np.clip(crop.astype(np.float32)*rng.uniform(.8,1.2)+rng.uniform(-12,12),0,255).astype(np.uint8)
            if rng.random()<.15:
                crop = cv2.GaussianBlur(crop,(3,3),rng.uniform(.3,.8))
            if rng.random()<.5:
                crop = np.ascontiguousarray(crop[:,::-1])
                points[:,0] = (SIZE-1)/SIZE-points[:,0]
                points,weights = points[FLIP],weights[FLIP]
                axis[:,0] = (SIZE-1)/SIZE-axis[:,0]
        x = np.ascontiguousarray(crop[:,:,::-1].transpose(2,0,1),dtype=np.float32)/255
        return x,points,weights,axis,axis_weights,np.float32(row['yaw_supported'] and abs(residual)<=55)


def loss_function(prediction, anchor, targets):
    import torch
    import torch.nn.functional as F
    points, weights, axis, axis_weights, yaw_supported = targets
    xy, _, _, heat, _ = prediction
    target = torch.cat([points,axis],1)
    mask = torch.cat([weights,axis_weights],1)
    n = int(math.sqrt(heat.shape[-1]))
    ce = (-(heat_targets(target,n)*heat.log_softmax(-1)).sum(-1)*mask).sum()/mask.sum().clamp(min=1)
    coord = (F.smooth_l1_loss(xy,target,beta=.02,reduction='none').sum(-1)*mask).sum()/mask.sum().clamp(min=1)
    # Nose offset divided by eye span is the actual downstream decision cue.
    # Supervise it only when all three teacher points have independent support.
    def yaw(p):
        a=p[:,1]-p[:,0];d=p[:,2]-(p[:,0]+p[:,1])/2
        # All eligible crops are upright; the convention matches face_geometry.
        return (a*d).sum(-1)/(a*a).sum(-1).clamp(min=1e-5)/1.05
    yw = yaw_supported*weights[:,:3].amin(1)
    yl = (F.smooth_l1_loss(yaw(xy),yaw(points),beta=.05,reduction='none')*yw).sum()/yw.sum().clamp(min=1)
    # Prevent unsupported points/rolls being rewritten by sparse pseudo labels.
    keep = 1-mask
    anchor_loss = (F.smooth_l1_loss(xy,anchor[0],beta=.02,reduction='none').sum(-1)*keep).sum()/keep.sum().clamp(min=1)
    total = .2*ce+20*coord+.5*yl+10*anchor_loss
    return total


def aggregate(errors):
    if not errors:
        return {'samples':0}
    return {'samples':len(errors),'mean':float(np.mean(errors)),'median':float(np.median(errors)),
            'p95':float(np.percentile(errors,95))}


def evaluate(model,dataset,device,batch_size=32):
    import torch
    model.eval()
    rows=[]
    with torch.inference_mode():
        for start in range(0,len(dataset),batch_size):
            samples=[dataset[i] for i in range(start,min(start+batch_size,len(dataset)))]
            x=torch.from_numpy(np.stack([s[0] for s in samples])).to(device)
            xy,logits,box,_,turn=model(x)
            xy=xy.cpu().numpy()
            for j,(sample,pred) in enumerate(zip(samples,xy)):
                row=dataset.rows[start+j]
                target,mask=sample[1],sample[2]>0
                iod=max(float(np.linalg.norm(target[1]-target[0])),1e-6)
                error=float(np.linalg.norm(pred[:6][mask]-target[mask],axis=1).mean()/iod)
                yaw_error=abs(float(yaw_numpy(pred)-yaw_numpy(target))) if row['yaw_supported'] else None
                rows.append({'id':row['id'],'video_id':row['video_id'],'bin':row['bin'],
                             'nme':error,'yaw_error':yaw_error,'teacher_yaw':row['yaw'],
                             'predicted_yaw':float(yaw_numpy(pred))})
    result={'metric_kind':'held-out teacher agreement, not human-annotated accuracy','samples':len(rows),
            'nme':aggregate([r['nme'] for r in rows]),
            'yaw_mae':aggregate([r['yaw_error'] for r in rows if r['yaw_error'] is not None]),
            'by_pose':{},'by_video':{}}
    for field,out in [('bin','by_pose'),('video_id','by_video')]:
        for value in sorted(set(r[field] for r in rows)):
            selected=[r for r in rows if r[field]==value]
            result[out][value]={'nme':aggregate([r['nme'] for r in selected]),
                               'yaw_mae':aggregate([r['yaw_error'] for r in selected if r['yaw_error'] is not None])}
    return result,rows


def validation_score(metrics):
    # Each video counts equally; thousands of adjacent frames from one dog
    # cannot dominate checkpoint selection. No test result is accessed here.
    scores=[]
    for row in metrics['by_video'].values():
        scores.append(row['nme']['mean']+row['yaw_mae'].get('mean',0))
    if not scores:
        raise ValueError('No usable validation samples; cannot select a checkpoint')
    return float(np.mean(scores))


def validation_guard(baseline, candidate):
    """A lower combined score is not permission to regress localization.

    This is only a screening guard for another evaluation round, never an
    automatic production-activation decision or a claim of ground-truth accuracy.
    """
    reasons=[]
    for name in ('nme','yaw_mae'):
        if candidate[name].get('mean',float('inf')) > baseline[name].get('mean',float('inf')):
            reasons.append(f'validation_{name}_regressed')
    for view in ('left','right'):
        if candidate['by_pose'].get(view,{}).get('yaw_mae',{}).get('samples',0)==0:
            reasons.append(f'no_supported_{view}_validation')
    return {'passed':not reasons,'reasons':reasons,'automatic_activation':False}


def export_checkpoint(weights,target,example_crops):
    import onnx
    import onnxruntime as ort
    import torch
    model,checkpoint=load_checkpoint(weights)
    model.eval()
    class Export(torch.nn.Module):
        def __init__(self,m):
            super().__init__();self.m=m
        def forward(self,x):
            return packed(self.m,x)
    target=Path(target)
    torch.onnx.export(Export(model),torch.zeros(1,3,SIZE,SIZE),str(target),opset_version=17,
                      input_names=['images'],output_names=['landmarks'],dynamo=False)
    onnx.checker.check_model(onnx.load(target))
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(target),options,providers=['CPUExecutionProvider'])
    errors=[]
    with torch.inference_mode():
        for x in example_crops:
            want=packed(model,torch.from_numpy(x[None])).numpy()
            got=session.run(None,{'images':x[None]})[0]
            np.testing.assert_allclose(got,want,rtol=1e-5,atol=1e-3)
            errors.append(float(np.max(np.abs(got-want))))
    metadata=dict(checkpoint['source_metadata'])
    metadata.update({'sha256':sha256(target),'source_weights':str(Path(weights).resolve()),
                     'training':checkpoint.get('training',{}),'status':'experimental_not_promoted',
                     'export_parity':{'real_crops':len(errors),'max_abs_error':max(errors)}})
    target.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    return metadata


def train(args):
    import torch
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if args.device=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but not available')
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    out=Path(args.out)
    if out.exists():
        raise FileExistsError(f'Use a fresh training run: {out}')
    rows,metadata=load_rows(args.data)
    train_rows=[r for r in rows if r['split']=='train']
    val_rows=[r for r in rows if r['split']=='val']
    if not train_rows or not val_rows:
        raise ValueError('Training and validation both need accepted samples')
    out.mkdir(parents=True)
    trainset=VideoCrops(args.data,train_rows,True,args.seed)
    valset=VideoCrops(args.data,val_rows,False,args.seed)
    model,checkpoint=load_checkpoint(args.weights,args.device)
    anchor,_=load_checkpoint(args.weights,args.device)
    anchor.eval()
    for p in anchor.parameters():
        p.requires_grad_(False)
    params=[]
    for name,p in model.named_parameters():
        active=name.startswith(('lat','fuse','heat'))
        p.requires_grad_(active)
        if active:params.append(p)
    optimizer=torch.optim.AdamW(params,lr=args.lr,weight_decay=1e-4)
    counts=Counter((r['video_id'],r['bin']) for r in train_rows)
    weights=[1/counts[(r['video_id'],r['bin'])] for r in train_rows]
    sampler=torch.utils.data.WeightedRandomSampler(weights,len(train_rows)*args.repeats,replacement=True)
    loader=torch.utils.data.DataLoader(trainset,batch_size=args.batch,sampler=sampler,num_workers=0)
    baseline,_=evaluate(model,valset,args.device)
    best_score=validation_score(baseline)
    run={'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
         'dataset_sha256':metadata['labels_sha256'],'teacher_sha256':metadata['teacher_sha256'],
         'initial_checkpoint_sha256':sha256(args.weights),
         'train_samples':len(train_rows),'val_samples':len(val_rows),
         'trainable_parameters':sum(p.numel() for p in params),
         'coverage':{split:{view:{'samples':sum(r['split']==split and r['bin']==view for r in rows),
                                 'yaw_supported':sum(r['split']==split and r['bin']==view and r['yaw_supported'] for r in rows)}
                            for view in ('front','left','right')} for split in ('train','val','test')},
         'frozen':'backbone, presence, visibility, box; runtime unchanged',
         'baseline_validation':baseline,'epochs':[]}
    def save(path,epoch):
        value={**checkpoint,'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
               'training':{'recipe':'video_distillation_fpn_v1','epoch':epoch,
                           'dataset_sha256':metadata['labels_sha256'],'teacher_sha256':metadata['teacher_sha256'],
                           'seed':args.seed,'frozen':run['frozen'],'parent_checkpoint_sha256':run['initial_checkpoint_sha256']}}
        torch.save(value,path)
    save(out/'best.pt',0)
    print(json.dumps({'baseline':baseline,'train_samples':len(train_rows),'trainable_parameters':run['trainable_parameters']}),flush=True)
    for epoch in range(1,args.epochs+1):
        started=time.perf_counter();total=0;steps=0
        model.train()
        for batch in loader:
            x,*targets=[value.to(args.device) for value in batch]
            with torch.no_grad():
                reference=anchor(x)
            prediction=model(x)
            loss=loss_function(prediction,reference,targets)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite loss; candidate not exported')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(params,1.0);optimizer.step()
            total+=float(loss.detach());steps+=1
        metrics,_=evaluate(model,valset,args.device)
        score=validation_score(metrics)
        selected=score<best_score
        if selected:
            best_score=score;save(out/'best.pt',epoch)
        log={'epoch':epoch,'loss':total/max(steps,1),'validation_score':score,
             'selected':selected,'seconds':time.perf_counter()-started,'validation':metrics}
        run['epochs'].append(log)
        (out/'training.json').write_text(json.dumps(run,indent=2),encoding='utf8')
        print(json.dumps({k:v for k,v in log.items() if k!='validation'}),flush=True)
    save(out/'last.pt',args.epochs)
    candidate,selected=load_checkpoint(out/'best.pt',args.device)
    comparison={'selected_epoch':selected['training']['epoch'],'status':'experimental_not_promoted',
                'limitations':['Teacher labels are not independent ground truth','Video pilot does not establish cross-breed production accuracy'],
                'splits':{}}
    for split in ('val','test'):
        dataset=VideoCrops(args.data,[r for r in rows if r['split']==split],False,args.seed)
        before,before_rows=evaluate(anchor,dataset,args.device)
        after,after_rows=evaluate(candidate,dataset,args.device)
        comparison['splits'][split]={'baseline':before,'candidate':after}
        (out/f'predictions_{split}.json').write_text(json.dumps({'baseline':before_rows,'candidate':after_rows},indent=2),encoding='utf8')
    comparison['validation_guard']=validation_guard(comparison['splits']['val']['baseline'],comparison['splits']['val']['candidate'])
    (out/'comparison.json').write_text(json.dumps(comparison,indent=2),encoding='utf8')
    # Free training allocations before loading CPU ORT for export parity.
    del model,anchor,candidate,optimizer
    if args.device=='cuda':torch.cuda.empty_cache()
    export_checkpoint(out/'best.pt',out/'dog_face_landmarks.onnx',[valset[i][0] for i in range(min(16,len(valset)))])
    print(json.dumps(comparison),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True,type=Path)
    parser.add_argument('--weights',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--epochs',type=int,default=12)
    parser.add_argument('--repeats',type=int,default=4)
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--lr',type=float,default=5e-5)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--seed',type=int,default=20261113)
    args=parser.parse_args()
    if min(args.epochs,args.repeats,args.batch,args.threads)<=0 or args.lr<=0:
        parser.error('Training parameters must be positive')
    train(args)


if __name__=='__main__':
    main()
