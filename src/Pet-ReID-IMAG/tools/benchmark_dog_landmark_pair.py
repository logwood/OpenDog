"""Serial, alternating baseline/candidate latency on identical real 128px crops.

Includes RGB conversion, synchronous inference (GPU transfers included), and
landmark decoding. This is model-call latency, not end-to-end capture FPS.
Use benchmark_dog_capture_v1.py with explicit refine.landmarks for video replay.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dog_capture.engine.detectors import CropLandmarker
from benchmark_dog_capture_v1 import preload_cuda, quantiles
from train_dog_video_landmarks import VideoCrops, load_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',required=True,type=Path)
    parser.add_argument('--candidate',required=True,type=Path)
    parser.add_argument('--data',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--providers',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--cuda-graph',action='store_true')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--rounds',type=int,default=4)
    parser.add_argument('--runs',type=int,default=150)
    args=parser.parse_args()
    if min(args.threads,args.rounds,args.runs)<=0:
        parser.error('threads/rounds/runs must be positive')
    if args.output.exists():
        parser.error('Use a fresh output file')
    if args.cuda_graph and args.providers!='cuda':
        parser.error('--cuda-graph requires CUDA')
    if args.providers=='cuda':
        preload_cuda()
    cv2.setNumThreads(1)
    rows,metadata=load_rows(args.data)
    dataset=VideoCrops(args.data,[r for r in rows if r['split']=='val'])
    if not len(dataset):
        parser.error('Need real validation crops')
    picks=np.linspace(0,len(dataset)-1,min(32,len(dataset))).round().astype(int)
    crops=[np.ascontiguousarray((dataset[int(i)][0].transpose(1,2,0)[:,:,::-1]*255).round(),dtype=np.uint8) for i in picks]
    models={name:CropLandmarker(path,providers=args.providers,threads=args.threads,
                               thread_spinning=False,cuda_graph=args.cuda_graph)
            for name,path in [('baseline',args.baseline),('candidate',args.candidate)]}
    expected='CUDAExecutionProvider' if args.providers=='cuda' else 'CPUExecutionProvider'
    for model in models.values():
        if model.runtime['used'][0]!=expected or model.runtime['fallback_errors']:
            raise RuntimeError('Requested provider was not used without fallback')
        for i in range(30):
            model.detect(crops[i%len(crops)])
        if args.cuda_graph and not model.runtime['cuda_graph']['active']:
            raise RuntimeError('CUDA Graph was requested but fell back')
    raw={name:[] for name in models};rounds=[]
    for round_index in range(args.rounds):
        order=['baseline','candidate'] if round_index%2==0 else ['candidate','baseline']
        record={'round':round_index+1,'order':order}
        for name in order:
            values=[]
            for i in range(args.runs):
                frame=crops[i%len(crops)]
                before=time.perf_counter_ns();models[name].detect(frame)
                values.append((time.perf_counter_ns()-before)/1e6)
            raw[name].extend(values);record[name]=quantiles(values)
        rounds.append(record)
    result={'metric':'one canonical crop model call; preprocessing + synchronized inference + decode',
            'provider':expected,'threads':args.threads,'thread_spinning':False,'cuda_graph':args.cuda_graph,
            'warmup_per_model':30,'samples':len(crops),'dataset_sha256':metadata['labels_sha256'],
            'models':{k:v.describe() for k,v in models.items()},'rounds':rounds,
            'summary':{k:quantiles(v) for k,v in raw.items()}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':
    main()
