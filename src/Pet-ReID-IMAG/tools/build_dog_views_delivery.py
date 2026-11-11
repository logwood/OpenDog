"""Package both a model-free integration SDK and the complete desktop evidence."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dog_capture.cli import ROOT
from dog_capture.models import ASSETS, file_hash


def finish(destination):
    records = {p.relative_to(destination).as_posix(): {'bytes': p.stat().st_size, 'sha256': file_hash(p)}
               for p in sorted(destination.rglob('*')) if p.is_file()}
    (destination / 'MANIFEST.sha256.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf8')
    archive = destination.with_suffix('.zip')
    if archive.exists():
        raise FileExistsError(archive)
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        for p in sorted(destination.rglob('*')):
            if p.is_file():
                z.write(p, Path(destination.name) / p.relative_to(destination))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    return {'directory': str(destination), 'archive': str(archive), 'bytes': archive.stat().st_size,
            'sha256': file_hash(archive), 'files': len(records)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(ROOT / 'delivery/dog_view_selection_v2'))
    parser.add_argument('--results', default=str(ROOT / 'artifacts/runs/dog_capture/v2_release'))
    args = parser.parse_args()
    full = Path(args.output).resolve()
    sdk = full.with_name(full.name + '_sdk')
    for path in (full, sdk):
        if path.exists() or path.with_suffix('.zip').exists():
            raise FileExistsError(path)
        path.mkdir(parents=True)
    def copy_file(relative, destination=full):
        source = ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for destination in (full, sdk):
        for p in (ROOT / 'src/Pet-ReID-IMAG/dog_capture').glob('*.py'):
            copy_file(p.relative_to(ROOT), destination)
        for p in ('requirements-dog-views-sdk.txt', 'docs/DOG_VIEW_SELECTION_CN.md', 'docs/DOG_VIEW_VALIDATION_CN.md',
                  'assets/dog_capture/view_targets.json', 'assets/dog_capture/NOTICE.txt',
                  'src/Pet-ReID-IMAG/tests/test_dog_capture.py', 'src/Pet-ReID-IMAG/tests/test_dog_views.py'):
            copy_file(p, destination)
        (destination / 'pytest.ini').write_text('[pytest]\npythonpath = src/Pet-ReID-IMAG\ntestpaths = src/Pet-ReID-IMAG/tests\n', encoding='utf8')
    (sdk / 'pyproject.toml').write_text('''[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
[project]
name = "dog-view-selection"
version = "0.2.0"
description = "Reuse existing dog face observations to select clear target views"
requires-python = ">=3.11"
dependencies = ["numpy>=1.26,<3", "opencv-python>=4.8,<6"]
[tool.setuptools]
package-dir = {"" = "src/Pet-ReID-IMAG"}
[tool.setuptools.packages.find]
where = ["src/Pet-ReID-IMAG"]
include = ["dog_capture*"]
''', encoding='utf8')
    (sdk / 'README.txt').write_text('指定位置选帧 SDK v0.2\n\n'
        '本包不含模型、视频和个人素材。推荐直接复用现有模型的 FaceDetection。\n'
        '在本目录运行 pip install .，或把 src/Pet-ReID-IMAG 加入 PYTHONPATH。\n'
        '已有 NumPy/OpenCV 环境可用 pip install . --no-deps。\n'
        'from dog_capture.views import TargetViewPipeline\n'
        'selector = TargetViewPipeline() 不加载任何神经网络。\n'
        '完整接法、输入契约、目标位置配置见 docs/DOG_VIEW_SELECTION_CN.md。\n'
        '只需 SDK 的 process/finalize/snapshot 接口；独立视频/摄像头工具与模型在完整验证包。\n', encoding='utf8')
    for name in ('requirements-dog-capture.txt', 'run-dog-views.cmd', 'run-dog-views-live.cmd',
                 'run-dog-capture.cmd', 'run-dog-capture-live.cmd', 'scripts/run-dog-capture.ps1',
                 'docs/DOG_CAPTURE_CN.md', 'docs/DOG_CAPTURE_VALIDATION_CN.md'):
        copy_file(name)
    for name in ('run_dog_capture.py', 'select_dog_views.py', 'live_dog_capture.py', 'live_dog_views.py',
                 'prepare_dog_capture_assets.py', 'validate_dog_capture.py', 'build_dog_capture_fixtures.py',
                 'verify_dog_capture_outputs.py', 'benchmark_dog_views.py', 'export_dog_view_model.py'):
        copy_file('src/Pet-ReID-IMAG/tools/' + name)
    for p in (ROOT / 'assets/dog_capture').rglob('*'):
        if p.is_file():
            copy_file(p.relative_to(ROOT))
    for manifest in ('sources.json', 'fixtures.json'):
        for entry in json.loads((ROOT / 'assets/dog_capture' / manifest).read_text(encoding='utf8')).values():
            assert file_hash(ROOT / entry['path']) == entry['sha256']
            copy_file(entry['path'])
    for name, (_, sha) in ASSETS.items():
        relative = Path('models/pretrained/dog_face_landmarks_v1') / name
        assert file_hash(ROOT / relative) == sha
        copy_file(relative)
    for p in (ROOT / 'models/pretrained/dog_face_landmarks_v2').glob('*'):
        if p.is_file():
            copy_file(p.relative_to(ROOT))
    results = Path(args.results)
    for name in ('validation', 'anyface_user', 'anyface_wolfhound', 'live_user', 'benchmark'):
        shutil.copytree(results / name, full / 'results' / name, ignore=shutil.ignore_patterns('*.log'))
    for name in ('user', 'wolfhound'):
        source = ROOT / 'artifacts/runs/dog_capture/v2_anyface' / (name + '_inputs')
        target = full / 'results/upstream_inputs' / name
        target.mkdir(parents=True)
        shutil.copy2(source / 'upstream_cost.json', target / 'upstream_cost.json')
        shutil.copy2(source / 'observations_auto_orientation.jsonl', target / 'observations.jsonl')
    for result in (full / 'results').rglob('result.json'):
        r = json.loads(result.read_text(encoding='utf8'))
        r['original_run_input'], r['output'] = r.get('input'), '.'
        r['paths_relative_to_result_file'] = True
        if r.get('recognition_input'):
            r['recognition_input']['image_path'] = r['selected']['front']['frame_file']
            r['recognition_input']['face_path'] = r['selected']['front']['face_file']
        result.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding='utf8')
    integrated = json.loads((results / 'anyface_wolfhound/result.json').read_text(encoding='utf8'))
    live = json.loads((results / 'live_user/result.json').read_text(encoding='utf8'))
    (full / 'README.txt').write_text('指定位置选帧 v0.2 完整验证包\n\n'
        '打开 START_HERE.html 查看你的图片、接口和实测结果。\n'
        '把新视频拖到 run-dog-views.cmd；实时窗口使用 run-dog-views-live.cmd。\n'
        '接入已有模型时推荐单独的 SDK 包，无需加载本包的后备模型。\n'
        '本包含用户测试视频，来源、许可、原图校验和测量范围见 assets 和 docs。\n', encoding='utf8')
    (full / 'START_HERE.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>指定位置选帧 v0.2</title>
<style>body{{font:17px/1.7 system-ui,"Microsoft Yahei",sans-serif;max-width:1000px;margin:38px auto;padding:0 24px;color:#173940;background:#eff4f4}}article{{background:white;padding:22px;border-radius:14px;margin:18px 0}}a{{color:#087e80}}img{{max-width:100%;height:320px;object-fit:contain}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}}code{{background:#e6eeee;padding:3px 6px}}</style>
<p>复用现有模型 · 每个位置一张 · 保留原始像素</p><h1>给比对模型的清晰图</h1>
<article><h2>推荐集成结构</h2><p>已有脸框与五点 → 位置与清晰度筛选 → 独立确认 → 每个位置一张原图。</p><p>选帧模块额外模型调用为 0；公开视频上的新增计算 P95 为 {integrated['performance']['compute_ms_p95']:.2f} ms，不含已有模型耗时。</p><p><a href="docs/DOG_VIEW_SELECTION_CN.md">接口与接法</a> · <a href="docs/DOG_VIEW_VALIDATION_CN.md">完整验证报告</a></p></article>
<div class="grid"><article><h2>复用现有 AnyFace</h2><img src="results/anyface_user/best_front_face.png"><p>你的视频：第 266 帧，8.867 秒。当前这条入口确认到正面。</p><p><a href="results/anyface_user/review.html">打开原图与结果</a></p></article>
<article><h2>独立视频入口</h2><img src="results/validation/user/best_left_face.png"><p>你的视频：已确认正面与左侧，右侧待补拍。</p><p><a href="results/validation/user/review.html">查看两个位置</a> · <a href="results/validation/user/result.json">比对输入</a></p></article></div>
<article><h2>速度与覆盖</h2><p>30 fps 实时回放：实际处理 {live['processing_fps']:.1f} fps，P95 延迟 {live['capture_to_result_ms_p95']:.1f} ms。</p><p><a href="results/anyface_wolfhound/review.html">复用五点收齐三个位置</a> · <a href="results/validation/wolfhound_panting/review.html">独立入口三个位置</a> · <a href="results/live_user/result.json">实时测量</a> · <a href="results/benchmark/benchmark.json">交替 A/B 计时</a></p><p>46 项测试，6 段自然视频和 4 个构造用例。方向比例没有标定为转角；手机耗时尚未测量。</p></article>
<article><h2>直接试新视频</h2><p>拖到 <code>run-dog-views.cmd</code> 做逐帧分析，或 <code>run-dog-views-live.cmd</code> 做实时回放。正面、左侧、右侧独立收集，无需按固定顺序转头。</p><p><a href="assets/dog_capture/sources.json">视频资产与来源</a> · <a href="assets/dog_capture/view_targets.json">位置配置</a></p></article></html>''', encoding='utf8')
    report = {'sdk': finish(sdk), 'complete': finish(full)}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
