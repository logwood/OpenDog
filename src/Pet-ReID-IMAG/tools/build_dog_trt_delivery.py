"""Assemble a local, hash-verified desktop TensorRT deliverable."""
import argparse
import html
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dog_capture.cli import ROOT
from dog_capture.models import ASSETS, file_hash
from build_dog_views_delivery import finish


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT / 'delivery/dog_view_selection_v3_trt')
    p.add_argument('--results', type=Path, default=ROOT / 'artifacts/runs/dog_capture/v3_final')
    args = p.parse_args()
    dest = args.output.resolve()
    if dest.exists() or dest.with_suffix('.zip').exists():
        raise FileExistsError(dest)
    dest.mkdir(parents=True)
    def copy(relative):
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    for file in (ROOT / 'src/Pet-ReID-IMAG/dog_capture').glob('*.py'):
        copy(file.relative_to(ROOT))
    for relative in ('requirements-dog-trt.txt', 'requirements-dog-trt-build.txt', 'requirements-dog-onnx-export.txt',
                     'requirements-dog-trt-test.txt',
                     'requirements-dog-capture.txt', 'requirements-dog-views-sdk.txt',
                     'scripts/setup-dog-trt.ps1', 'scripts/run-dog-trt.ps1', 'scripts/run-dog-capture.ps1',
                     'run-dog-trt.cmd', 'run-dog-trt-live.cmd', 'run-dog-views.cmd', 'run-dog-views-live.cmd',
                     'docs/DOG_TENSORRT_CN.md', 'docs/DOG_TENSORRT_VALIDATION_CN.md',
                     'docs/DOG_VIEW_SELECTION_CN.md', 'docs/DOG_VIEW_VALIDATION_CN.md'):
        copy(relative)
    for name in ('select_dog_views.py', 'live_dog_views.py', 'run_dog_capture.py', 'live_dog_capture.py',
                 'prepare_dog_trt.py', 'benchmark_dog_trt.py', 'validate_dog_trt.py', 'profile_dog_trt.py',
                 'bootstrap_dog_trt.py', 'create_dog_trt_runtime.py', 'configure_dog_trt_windows.py',
                 'prepare_dog_capture_assets.py', 'validate_dog_capture.py', 'build_dog_capture_fixtures.py',
                 'verify_dog_capture_outputs.py', 'export_dog_view_model.py'):
        copy(Path('src/Pet-ReID-IMAG/tools') / name)
    for name in ('test_dog_capture.py', 'test_dog_views.py', 'test_dog_trt.py'):
        copy(Path('src/Pet-ReID-IMAG/tests') / name)
    for file in (ROOT / 'assets/dog_capture').rglob('*'):
        if file.is_file():
            copy(file.relative_to(ROOT))
    for name in ASSETS:
        relative = Path('models/pretrained/dog_face_landmarks_v1') / name
        assert file_hash(ROOT / relative) == ASSETS[name][1]
        copy(relative)
    engines = ROOT / 'models/pretrained/dog_capture_trt_v3'
    for name in ('localizer', 'landmarks', 'detector'):
        meta = json.loads((engines / (name + '.json')).read_text(encoding='utf8'))
        for extension, key in (('.engine', 'engine_sha256'), ('.onnx', 'onnx_sha256')):
            assert file_hash(engines / (name + extension)) == meta[key]
            copy((engines / (name + extension)).relative_to(ROOT))
        copy((engines / (name + '.json')).relative_to(ROOT))
    for name in ('build.json', 'export.json', 'detector_contract.json', 'landmarks.raw.onnx'):
        copy((engines / name).relative_to(ROOT))
    for name in ('sources.json', 'fixtures.json'):
        for entry in json.loads((ROOT / 'assets/dog_capture' / name).read_text(encoding='utf8')).values():
            assert file_hash(ROOT / entry['path']) == entry['sha256']
            copy(entry['path'])
    shutil.copytree(args.results, dest / 'results', ignore=shutil.ignore_patterns('*.log', '*.npz', '__pycache__'))
    for path in (dest / 'results').rglob('result.json'):
        report = json.loads(path.read_text(encoding='utf8'))
        report['original_run_output'] = report.get('output')
        report['output'] = '.'
        report['paths_relative_to_result_file'] = True
        if report.get('recognition_input'):
            report['recognition_input']['image_path'] = report['selected']['front']['frame_file']
            report['recognition_input']['face_path'] = report['selected']['front']['face_file']
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    live = json.loads((args.results / 'comparison/live-user-trt/result.json').read_text(encoding='utf8'))
    user = json.loads((args.results / 'validation/user/result.json').read_text(encoding='utf8'))
    cards = ''.join(f'<article><h2>{html.escape(name)}</h2><img src="results/validation/user/{item["face_file"]}"><p>第 {item["frame_index"]} 帧 · {item["time_s"]:.3f} 秒</p><a href="results/validation/user/{item["frame_file"]}">原始完整帧</a></article>' for name, item in user['selected'].items())
    (dest / 'START_HERE.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>犬只指定位置选帧 · TensorRT</title>
<style>body{{font:17px/1.7 system-ui,"Microsoft Yahei",sans-serif;color:#173940;background:#edf4f4;max-width:1080px;margin:36px auto;padding:0 24px}}article{{background:white;padding:24px;border-radius:14px;margin:18px 0}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:20px}}img{{width:100%;height:300px;object-fit:contain}}a{{color:#08777b}}code{{background:#dcebea;padding:3px 6px}}</style>
<p>v0.3 · Windows / RTX 4060 Laptop</p><h1>选出正面与两侧的清晰原图</h1>
<article><h2>实时验证</h2><p>你的 30 fps 视频：处理 {live['frames_processed']} / {live['source_frames']} 帧，结果延迟 P95 <strong>{live['capture_to_result_ms_p95']:.2f} ms</strong>。</p><p>延迟从已解码帧提交到结果可用，不含传感器、解码与显示；完整测试范围见验证报告。</p><p><a href="docs/DOG_TENSORRT_VALIDATION_CN.md">性能与质量验证</a> · <a href="results/comparison.json">完整对照数据</a> · <a href="results/comparison/live-user-trt/result.json">实时原始记录</a></p></article>
<div class="grid">{cards}</div><p>缺少的位置：{html.escape(', '.join(user['missing_views']) or '无')}。只保存经过质量、时序和原帧复核的位置。</p>
<article><h2>直接试新视频</h2><p>首次执行 <code>scripts/setup-dog-trt.ps1</code>，然后将视频拖到 <code>run-dog-trt.cmd</code>；实时窗口用 <code>run-dog-trt-live.cmd</code>。</p><p><a href="docs/DOG_TENSORRT_CN.md">安装与运行</a> · <a href="docs/DOG_VIEW_SELECTION_CN.md">复用已有模型的接口</a></p></article>
<article><h2>检查结果</h2><p><a href="results/validation/user/review.html">你的视频选帧</a> · <a href="results/validation/wolfhound_panting/review.html">公开视频三个位置</a> · <a href="results/validation/validation.json">六段自然视频与四个构造用例</a> · <a href="results/pixel_checks.json">原始像素校验</a></p><p><a href="assets/dog_capture/NOTICE.txt">模型与素材说明</a> · <a href="MANIFEST.sha256.json">文件哈希清单</a></p></article></html>''', encoding='utf8')
    (dest / 'pytest.ini').write_text('[pytest]\npythonpath = src/Pet-ReID-IMAG\ntestpaths = src/Pet-ReID-IMAG/tests\n', encoding='utf8')
    (dest / 'README.txt').write_text('打开 START_HERE.html 查看结果、实测与运行方式。\n首次运行 scripts/setup-dog-trt.ps1 安装依赖。\n本包含用户私有测试素材，仅供本项目本地验证。\n', encoding='utf8')
    print(json.dumps(finish(dest), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
