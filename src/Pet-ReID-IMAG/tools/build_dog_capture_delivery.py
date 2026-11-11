"""Package only this feature, pinned models, attributed assets and verified runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dog_capture.cli import ROOT
from dog_capture.models import ASSETS, file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(ROOT / 'delivery/dog_capture_v1'))
    parser.add_argument('--validation', default='artifacts/runs/dog_capture/release_validation')
    parser.add_argument('--live', default='artifacts/runs/dog_capture/release_live_user')
    parser.add_argument('--fixtures', default='artifacts/runs/dog_capture/release_fixtures')
    args = parser.parse_args()
    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    def copy_file(path):
        source = ROOT / path
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for name in ['requirements-dog-capture.txt', 'run-dog-capture.cmd', 'run-dog-capture-live.cmd',
                 'scripts/run-dog-capture.ps1', 'docs/DOG_CAPTURE_CN.md', 'docs/DOG_CAPTURE_VALIDATION_CN.md',
                 'src/Pet-ReID-IMAG/tests/test_dog_capture.py']:
        copy_file(name)
    for file in (ROOT / 'src/Pet-ReID-IMAG/dog_capture').glob('*.py'):
        copy_file(file.relative_to(ROOT))
    for name in ['run_dog_capture.py', 'live_dog_capture.py', 'prepare_dog_capture_assets.py',
                 'validate_dog_capture.py', 'build_dog_capture_fixtures.py', 'verify_dog_capture_outputs.py']:
        copy_file('src/Pet-ReID-IMAG/tools/' + name)
    for file in (ROOT / 'assets/dog_capture').iterdir():
        if file.is_file():
            copy_file(file.relative_to(ROOT))
    for manifest_name in ('sources.json', 'fixtures.json'):
        manifest = json.loads((ROOT / 'assets/dog_capture' / manifest_name).read_text(encoding='utf8'))
        for asset in manifest.values():
            if file_hash(ROOT / asset['path']) != asset['sha256']:
                raise ValueError(f"Asset changed: {asset['path']}")
            copy_file(asset['path'])
    for name, (_, digest) in ASSETS.items():
        relative = Path('models/pretrained/dog_face_landmarks_v1') / name
        if file_hash(ROOT / relative) != digest:
            raise ValueError(f'Model changed: {name}')
        copy_file(relative)
    for name, source in [('validation', args.validation), ('live_user', args.live), ('fixtures', args.fixtures)]:
        shutil.copytree(ROOT / source, destination / 'results' / name, ignore=shutil.ignore_patterns('*.log'))
    for result_file in (destination / 'results').rglob('result.json'):
        report = json.loads(result_file.read_text(encoding='utf8'))
        report['original_run_input'] = report.get('input')
        report['paths_relative_to_result_file'] = True
        report['output'] = '.'
        if report.get('recognition_input'):
            report['recognition_input']['image_path'] = report['selected']['front']['frame_file']
            report['recognition_input']['face_path'] = report['selected']['front']['face_file']
        result_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    (destination / 'pytest.ini').write_text('[pytest]\npythonpath = src/Pet-ReID-IMAG\ntestpaths = src/Pet-ReID-IMAG/tests/test_dog_capture.py\n', encoding='utf8')
    (destination / 'README.txt').write_text(
        '犬只转头采集与清晰帧选择 v0.1\n\n'
        '1. 打开 START_HERE.html 查看结果、主图及说明。\n'
        '2. 把新视频拖到 run-dog-capture.cmd 做逐帧分析。\n'
        '3. 拖到 run-dog-capture-live.cmd 做实时窗口回放。\n'
        '首次运行自动创建 Python 3.12 隔离环境并安装依赖；本包已含模型和测试素材。\n'
        '这是桌面验证版；许可、实测范围与未完成的手机验证见 docs。\n'
        '用户视频属于私人测试素材；请勿把整个包公开上传。\n', encoding='utf8')
    (destination / 'START_HERE.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>犬只采集验证版</title><style>body{font:17px/1.7 system-ui,"Microsoft Yahei",sans-serif;background:#f4f6f8;color:#1e293b;max-width:900px;margin:40px auto;padding:0 24px}article{background:white;padding:24px;border-radius:14px;margin:18px 0}a{color:#0f766e}img{max-width:100%;max-height:420px}code{background:#e2e8f0;padding:3px 6px}</style>
<h1>犬只转头与清晰帧采集</h1><p>电脑端验证版 · 本地处理 · 原始视频已归档</p>
<article><h2>你的视频</h2><p><a href="results/validation/user/review.html">打开完整标注回放与选图</a></p><img src="results/validation/user/best_front_face.png"><p><a href="results/live_user/result.json">实时回放测量结果</a></p></article>
<article><h2>公开样本</h2><p><a href="results/validation/wolfhound_panting/review.html">完整左右转头流程</a> · <a href="results/validation/labrador_barking/review.html">正面样本</a> · <a href="results/validation/lurcher_lick/review.html">舔食与遮挡</a></p><p><a href="results/validation/shepherd_apple/review.html">俯头样本</a> · <a href="results/validation/dog_whimpering/review.html">远景与栅栏样本</a></p></article>
<article><h2>运行与验证</h2><p>将视频拖到 <code>run-dog-capture.cmd</code>；实时窗口使用 <code>run-dog-capture-live.cmd</code>。</p><p><a href="docs/DOG_CAPTURE_CN.md">使用与接口说明</a> · <a href="docs/DOG_CAPTURE_VALIDATION_CN.md">验收报告与适用范围</a> · <a href="assets/dog_capture/sources.json">资产及来源清单</a></p><p>动作完成与主图选取单独记录；手机端和跨犬种上线可靠性仍需验证。</p></article></html>''', encoding='utf8')
    records = {p.relative_to(destination).as_posix(): {'bytes': p.stat().st_size, 'sha256': file_hash(p)}
               for p in sorted(destination.rglob('*')) if p.is_file()}
    (destination / 'MANIFEST.sha256.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf8')
    archive = destination.with_suffix('.zip')
    if archive.exists():
        raise FileExistsError(archive)
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        for path in sorted(destination.rglob('*')):
            if path.is_file():
                z.write(path, Path(destination.name) / path.relative_to(destination))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('Delivery archive checksum test failed')
    print(json.dumps({'directory': str(destination), 'archive': str(archive), 'archive_bytes': archive.stat().st_size,
                      'archive_sha256': file_hash(archive), 'files': len(records)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
