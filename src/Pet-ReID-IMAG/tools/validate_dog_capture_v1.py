"""Reproducible real-model video regression; records coverage without claiming held-out accuracy."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def acceptance(name, selection):
    specification = json.loads((ROOT / 'assets/dog_capture/v1_reference/acceptance.json').read_text(encoding='utf8'))
    rule = specification['cases'].get(name)
    if rule is None:
        return {'checked': False, 'passed': None, 'reasons': []}
    views = selection['views']
    reasons = [f'missing:{v}' for v in rule.get('required_views', []) if v not in views]
    reasons += [f'forbidden_view:{v}' for v in rule.get('forbidden_views', []) if v in views]
    if rule.get('forbid_any_capture') and views:
        reasons.append('unacceptable_image_selected')
    for shots in views.values():
        reasons += [f'rejected_frame:{s["frame_index"]}' for s in shots if s['frame_index'] in rule.get('rejected_frames', [])]
    return {'checked': True, 'passed': not reasons, 'reasons': reasons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--case', action='append', help='case name; default all archived videos and fixtures')
    parser.add_argument('--enforce-acceptance', action='store_true', help='exit 3 if model release criteria fail')
    args = parser.parse_args()
    cases = {'user': ROOT / 'data/dog_capture/user/2611101816341481.mp4'}
    for folder in ('public', 'fixtures'):
        cases.update({p.stem: p for p in sorted((ROOT / 'data/dog_capture' / folder).glob('*'))
                      if p.suffix.lower() in ('.mp4', '.ogv', '.webm')})
    wanted = args.case or list(cases)
    args.output.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, 'PYTHONPATH': str(ROOT / 'src/Pet-ReID-IMAG')}
    summary = []
    for name in wanted:
        source = cases[name]
        if not source.is_file():
            raise FileNotFoundError(source)
        output = args.output / name
        if output.exists():
            raise ValueError(f'Use a fresh output directory; {output} already exists')
        command = [sys.executable, '-m', 'dog_capture', 'run', str(source), '--output', str(output),
                   '--providers', 'cpu', '--threads', str(args.threads), '--mode', 'best', '--quiet']
        with (args.output / f'{name}.log').open('w', encoding='utf8') as log:
            run = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if run.returncode not in (0, 2):
            raise RuntimeError(f'{name} failed with {run.returncode}; see its log')
        report = json.loads((output / 'result.json').read_text(encoding='utf8'))
        row = {'case': name, 'source': source.relative_to(ROOT).as_posix(), 'exit_code': run.returncode,
               'selection': report['selection'], 'stats': report['stats'], 'guidance_counts': report['guidance_counts'],
               'models': report['models'], 'review': f'{name}/review.html'}
        row['acceptance'] = acceptance(name, report['selection'])
        summary.append(row)
        (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
        print(f'{name}: {list(row["selection"]["views"])}; missing={row["selection"]["missing_views"]}', flush=True)
        if name in ('empty_scene_fixture', 'blurred_front_fixture') and row['selection']['views']:
            raise AssertionError(f'{name}: unacceptable image was selected')
    print(args.output / 'summary.json')
    return 3 if args.enforce_acceptance and any(r['acceptance']['passed'] is False for r in summary) else 0


if __name__ == '__main__':
    raise SystemExit(main())
