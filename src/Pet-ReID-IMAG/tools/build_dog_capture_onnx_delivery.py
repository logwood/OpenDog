"""Build a self-contained source/model/sample bundle for the v1 ONNX capture integration.

No environment, user gallery, credentials, unrelated Re-ID source or recognition weights are copied.
Existing destinations are never overwritten. Requires an already built H5 and validation results.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'delivery/dog_capture_onnx_v1')
    parser.add_argument('--evidence', type=Path, default=ROOT / 'artifacts/runs/dog_capture/v1_integration')
    args = parser.parse_args()
    out, evidence = args.output.resolve(), args.evidence.resolve()
    archive = out.with_suffix('.zip')
    if out.exists() or archive.exists():
        raise FileExistsError(f'Use a fresh destination: {out}')
    required = [ROOT / 'models/pretrained/dog_capture_v1/dog_face_pose.onnx',
                ROOT / 'models/pretrained/dog_capture_v1/yolox_nano.onnx',
                ROOT / 'src/Pet-ReID-IMAG/frontend/dog-capture/packages/h5/dist/index.html',
                evidence / 'browser-reid-final/summary.json']
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    entries = json.loads((ROOT / 'assets/dog_capture/model_zoo_v1.json').read_text(encoding='utf8'))['models']
    for name in ('dog_face_pose', 'yolox_nano'):
        entry = next(e for e in entries if e['name'] == name)
        assert digest(ROOT / 'models/pretrained/dog_capture_v1' / entry['file']) == entry['sha256']
    out.mkdir(parents=True)

    def copy(source, relative=None):
        source = Path(source)
        relative = relative or source.relative_to(ROOT)
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    def tree(source, relative, *, exclude=()):
        for path in sorted(source.rglob('*')):
            parts = path.relative_to(source).parts
            if path.is_file() and not any(p in exclude for p in parts) and path.suffix not in ('.pyc', '.tsbuildinfo'):
                copy(path, Path(relative) / path.relative_to(source))

    tree(ROOT / 'src/Pet-ReID-IMAG/dog_capture', 'src/Pet-ReID-IMAG/dog_capture', exclude=('__pycache__',))
    tree(ROOT / 'src/Pet-ReID-IMAG/frontend/dog-capture', 'src/Pet-ReID-IMAG/frontend/dog-capture',
         exclude=('node_modules', 'public', 'results', 'media', 'test-results', '__pycache__'))
    tree(ROOT / 'assets/dog_capture', 'assets/dog_capture')
    tree(ROOT / 'models/pretrained/dog_capture_v1', 'models/pretrained/dog_capture_v1', exclude=('__pycache__',))
    tree(ROOT / 'data/dog_capture', 'data/dog_capture', exclude=('raw', 'cache'))
    for path in ('requirements-dog-capture-v1.txt', 'requirements-dog-capture-service.txt',
                 'requirements-dog-capture-validation.txt', 'dog-capture.cmd',
                 'scripts/setup-dog-capture-v1.ps1', 'scripts/start-dog-capture-v1.ps1',
                 'scripts/dog-capture.ps1', 'scripts/dog-capture.sh',
                 'src/Pet-ReID-IMAG/tools/serve_dog_capture.py',
                 'src/Pet-ReID-IMAG/tools/validate_dog_capture_v1.py',
                 'src/Pet-ReID-IMAG/tools/build_dog_capture_onnx_delivery.py',
                 'docs/DOG_CAPTURE_ENGINE_CN.md', 'docs/DOG_CAPTURE_ENGINE_VALIDATION_CN.md',
                 'docs/DOG_CAPTURE_PROTOCOL_CN.md', 'docs/DOG_CAPTURE_V1_INTEGRATION_CN.md'):
        copy(ROOT / path)
    tests = ROOT / 'src/Pet-ReID-IMAG/tests'
    for name in ('test_dog_engine.py', 'test_dog_engine_io.py', 'test_dog_engine_onnx.py',
                 'test_dog_capture_service.py', 'dog_engine_synthetic.py'):
        copy(tests / name)
    for name in ('regression', 'user_quality', 'browser-reid-final', 'browser-final'):
        tree(evidence / name, f'results/{name}')
    for path in evidence.glob('*.log'):
        copy(path, Path('results/logs') / path.name)
    for name in ('front.png', 'side.png', 'source.json'):
        copy(evidence / 'e2e-media' / name, f'data/dog_capture/e2e/{name}')
    copy(evidence / 'golden/ref.json', 'data/dog_capture/e2e/ref.json')
    copy(evidence / 'golden/frame.bgr', 'data/dog_capture/e2e/frame.bgr')
    shutil.copy2(out / 'docs/DOG_CAPTURE_V1_INTEGRATION_CN.md', out / 'README_CN.md')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    rows = json.loads((evidence / 'regression/summary.json').read_text(encoding='utf8'))
    links = ''.join(f'<li><a href="results/regression/{html.escape(r["case"])}/review.html">'
                    f'{html.escape(r["case"])}：{html.escape(", ".join(r["selection"]["views"]) or "未采到")}</a></li>'
                    for r in rows)
    page = '''<!doctype html><meta charset="utf-8"><title>犬脸采集 v1 · 本机集成</title>
<style>body{font:17px/1.7 system-ui;max-width:960px;margin:45px auto;padding:0 24px;color:#182d34;background:#f5f8f7}
h1{font-size:32px}pre{background:#e5edeb;padding:18px;overflow:auto}a{color:#126256}.note{border-left:4px solid #c38728;padding:12px;background:#fff5dc}img{max-width:360px;border-radius:12px}</style>
<h1>犬脸采集 v1 · 本机集成</h1>
<p>包含用户提供的 320×320 ONNX、6 点配置、YOLOX-nano、原生引擎、SDK、已构建 H5 和服务端协议。</p>
<p class="note">接入和真实 Re-ID 联调已通过。当前权重在用户视频上的方向与选图验收未通过，本包保留失败记录；不作为已达标的替代模型。</p>
<h2>运行</h2><p>Python 3.11+（本机验证 3.12）；首次安装依赖需要联网。已包含模型和 H5 构建产物。</p>
<pre>powershell -ExecutionPolicy Bypass -File scripts/setup-dog-capture-v1.ps1
dog-capture.cmd data/dog_capture/user/2611101816341481.mp4 --config assets/dog_capture/v1_reference/capture_quality.json --mode best
.tmp/dog-capture-v1-env/Scripts/python.exe -m pip install -r requirements-dog-capture-service.txt
powershell -ExecutionPolicy Bypass -File scripts/start-dog-capture-v1.ps1</pre>
<p>浏览器打开 http://127.0.0.1:8765/capture/。独立服务只负责采集、复检及提供原图；建档与比对通过现有 Pet ReID 服务挂载。</p>
<p><a href="README_CN.md">完整接入与复现说明</a> · <a href="assets/dog_capture/v1_reference/integration_report.json">验证摘要</a> · <a href="manifest.json">文件校验清单</a></p>
<h2>真实识别服务联调</h2><p>独立测试图库中的识别、两侧建档和重拍；采用照片生成的虚拟摄像头素材，只验证链路。</p>
<a href="results/browser-reid-final/identify_front-5-committed.png"><img src="results/browser-reid-final/identify_front-5-committed.png"></a>
<a href="results/browser-reid-final/enroll_sides-5-committed.png"><img src="results/browser-reid-final/enroll_sides-5-committed.png"></a>
<h2>逐帧回归结果</h2><p>以下是默认兼容配置的输出，包含未通过验收的案例；质量优先配置拒绝用户视频中那张低质量候选。</p><ul>''' + links + '</ul>'
    (out / 'START_HERE.html').write_text(page, encoding='utf8')
    inventory = [{'path': p.relative_to(out).as_posix(), 'bytes': p.stat().st_size, 'sha256': digest(p)}
                 for p in sorted(out.rglob('*')) if p.is_file()]
    manifest = {'source_head_at_packaging': commit, 'capture_base_commit': 'e2588193abd25e157d87aec87aec550133690cc2',
                'contents_sha256_are_authoritative': True, 'files': inventory}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                bundle.write(p, p.relative_to(out.parent).as_posix())
    (archive.with_suffix('.zip.sha256')).write_text(f'{digest(archive)}  {archive.name}\n', encoding='utf8')
    print(json.dumps({'directory': str(out), 'zip': str(archive), 'bytes': archive.stat().st_size,
                      'sha256': digest(archive), 'files': len(inventory)}, indent=2))


if __name__ == '__main__':
    main()
