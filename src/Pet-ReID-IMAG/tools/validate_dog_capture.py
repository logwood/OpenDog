"""Reproducible full-video checks and coarse manually reviewed regression points."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dog_capture.cli import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--reuse-existing', action='store_true', help='Read existing completed results instead of rerunning')
    parser.add_argument('--mode', choices=['challenge', 'views'], default='challenge')
    parser.add_argument('--device', choices=['auto', 'cpu', 'gpu', 'trt'], default='auto')
    parser.add_argument('--no-preview', action='store_true')
    parser.add_argument('--include-fixtures', action='store_true')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = json.loads((ROOT / 'assets/dog_capture/sources.json').read_text(encoding='utf8'))
    fixtures = json.loads((ROOT / 'assets/dog_capture/fixtures.json').read_text(encoding='utf8')) if args.include_fixtures else {}
    sources.update(fixtures)
    annotation_file = ROOT / 'assets/dog_capture/regression_labels.json'
    labels = json.loads(annotation_file.read_text(encoding='utf8'))
    rows, comparisons = [], []
    for name, asset in sources.items():
        key = 'user' if asset.get('source') == 'user-provided' else Path(name).stem
        run_dir = output / key
        result = run_dir / 'result.json'
        if not (args.reuse_existing and result.exists()):
            command = [sys.executable, str(ROOT / 'src/Pet-ReID-IMAG/tools/run_dog_capture.py'),
                       str(ROOT / asset['path']), '--output', str(run_dir), '--mode', args.mode, '--device', args.device]
            if args.no_preview:
                command.append('--no-preview')
            with (output / f'{key}-runtime.log').open('w', encoding='utf8') as runtime_log:
                subprocess.run(command, check=True, stderr=runtime_log)
        report = json.loads(result.read_text(encoding='utf8'))
        if name in fixtures:
            assert bool(report['recognition_input']) == asset['expected_primary_image'], (name, 'primary image')
            if args.mode == 'views':
                assert report['coverage_complete'] == (name == 'ordered_turn_fixture.mp4'), (name, 'coverage')
                assert not report['sequence_complete']
            else:
                assert report['sequence_complete'] == asset['expected_sequence_complete'], (name, 'sequence')
        records = {r['frame_index']: r for r in map(json.loads, (run_dir / 'frames.jsonl').read_text(encoding='utf8').splitlines())}
        counts = Counter()
        for annotation in labels['videos'].get(name, []):
            record = records.get(annotation['frame'])
            if record is None:
                counts['missing'] += 1
                continue
            expected = annotation['expected']
            predicted = record['tracking']['stable_state']
            if expected == 'reject':
                outcome = 'correct_rejection' if not record['quality']['accepted'] else 'false_quality_accept'
            elif predicted == 'unknown':
                outcome = 'abstained'
            else:
                outcome = 'correct_direction' if predicted == expected else 'wrong_direction'
            counts[outcome] += 1
            comparisons.append({'video': name, 'frame': annotation['frame'], 'expected': expected,
                                'predicted': predicted, 'quality_accepted': record['quality']['accepted'], 'outcome': outcome})
        rows.append({'video': name, 'frames': report['frames_analyzed'], 'video_seconds': report['video_seconds'],
                     'analysis_fps': report['performance']['analysis_fps'],
                     'compute_ms_p95': report['performance']['compute_ms_p95'],
                     'sequence_complete': report['sequence_complete'],
                     'coverage_complete': report.get('coverage_complete'),
                     'missing_views': report.get('missing_views'),
                     'selected': {k: v['frame_index'] for k, v in report['selected'].items()},
                     'constructed_fixture': name in fixtures,
                     'regression_points': dict(counts)})
    aggregate = Counter(c['outcome'] for c in comparisons)
    summary = {'scope': labels['scope'], 'videos': rows, 'regression_totals': dict(aggregate),
               'regression_comparisons': comparisons,
               'notes': ['These manually reviewed coarse checkpoints are a regression aid, not an independent breed-level accuracy benchmark.',
                         'Unknown/abstained is counted separately, never counted as a correct head direction.']}
    (output / 'validation.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'videos': rows, 'regression_totals': dict(aggregate)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
