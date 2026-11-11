"""Layer profiling for diagnosis only; profiler overhead is not a benchmark."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dog_capture.trt import TrtModel, load_cuda_dlls


def main():
    p = argparse.ArgumentParser()
    p.add_argument('engine', type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    load_cuda_dlls()
    import tensorrt as trt
    class Profiler(trt.IProfiler):
        def __init__(self):
            super().__init__()
            self.times = defaultdict(list)
        def report_layer_time(self, name, milliseconds):
            self.times[name].append(milliseconds)
    with TrtModel(args.engine, use_cuda_graph=False) as model:
        profiler = Profiler()
        model.context.profiler = profiler
        for _ in range(10):
            model(np.full(model.input['shape'], .5, model.input['dtype']))
        rows = sorted([{'layer': name, 'mean_ms': float(np.mean(values)), 'n': len(values)}
                       for name, values in profiler.times.items()], key=lambda x: -x['mean_ms'])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2), encoding='utf8')
        print(json.dumps(rows[:12], indent=2))


if __name__ == '__main__':
    main()
