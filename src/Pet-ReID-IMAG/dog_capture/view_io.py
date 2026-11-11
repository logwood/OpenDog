"""Optional disk adapters; not imported by the hot-path selector."""
import html
import json

from .observations import FaceObservation
from .views import DEFAULT_TARGETS, ViewTarget


def load_targets(path=None):
    if not path:
        return DEFAULT_TARGETS
    with open(path, encoding='utf8') as stream:
        return tuple(ViewTarget(**value) for value in json.load(stream))


class ObservationReader:
    """Streaming ordered JSONL; wrapper: {frame_index, observation: {...}|null}."""
    def __init__(self, path):
        self.stream = open(path, encoding='utf8')
        self.current = None
        self.previous_index = -1

    def get(self, frame_index):
        while self.current is None or self.current['frame_index'] < frame_index:
            line = self.stream.readline()
            if not line:
                return None
            value = json.loads(line)
            index = value['frame_index']
            if not isinstance(index, int) or index <= self.previous_index:
                raise ValueError('Upstream JSONL indices must be strictly increasing integers')
            self.previous_index, self.current = index, value
        if self.current['frame_index'] != frame_index or self.current.get('observation') is None:
            return None
        return FaceObservation.from_dict(self.current['observation'])

    def close(self):
        self.stream.close()


def write_view_review(output, report):
    labels = {'front': '正面', 'left': '朝左', 'right': '朝右'}
    cards = []
    for target in report['view_configuration']['targets']:
        name = target['name']
        item = report['selected'].get(name)
        title = html.escape(labels.get(name, name))
        if not item:
            cards.append(f'<article><h2>{title}</h2><div class="empty">此段没有确认到合格图像</div><p>待补拍这个位置。</p></article>')
            continue
        q = item['quality']
        cards.append(f'<article><h2>{title}</h2><a href="{item["frame_file"]}"><img src="{item["face_file"]}" alt="{title}原始脸部裁剪"></a>'
                     f'<p>{item["time_s"]:.3f} 秒 · 第 {item["frame_index"]} 帧<br>位置线索 {item["pose"]["yaw_cue"]:+.3f} · 选帧分 {q["selection_score"]:.2f}</p>'
                     f'<p><a href="{item["frame_file"]}">原始整帧</a> · <a href="{item["face_file"]}">脸部裁剪</a></p></article>')
    perf = report['performance']
    video = f'<video controls preload="metadata" src="{report["preview_file"]}"></video>' if report['preview_file'] else ''
    body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>犬只指定位置选帧</title><style>body{{font:16px/1.7 system-ui,"Microsoft Yahei",sans-serif;background:#f2f5f6;color:#152f36;margin:0}}main{{max-width:1100px;margin:36px auto;padding:0 24px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:18px}}article,.stats{{background:white;padding:20px;border-radius:14px;margin:14px 0}}img{{width:100%;height:280px;object-fit:contain;background:#edf1f2}}.empty{{height:280px;display:grid;place-items:center;background:#edf1f2;color:#677b80}}a{{color:#087e80}}video{{width:100%;max-height:600px}}.muted{{color:#61777e;overflow-wrap:anywhere}}h1{{font-size:30px}}pre{{white-space:pre-wrap}}</style>
<main><p class="muted">指定位置 · 独立确认 · 原始像素</p><h1>给比对模型的几张清晰图</h1>
<p class="muted">{html.escape(report['input'])}</p><div class="stats">已收集 {len(report['selected'])}/{len(report['view_configuration']['targets'])} 个位置；无需按顺序转头或回正。<br>
处理 {report['frames_analyzed']} 帧，{perf['analysis_fps']:.1f} 帧/秒；单帧计算 P95 {perf['compute_ms_p95']:.1f} ms。<br>计算设备：{html.escape(perf['device'])}</div>
<div class="grid">{''.join(cards)}</div>{video}
<p>左右按转正后的脸部画面定义。位置线索是关键点几何比例，不代表角度；只比较指定范围内的清晰度。每张图都通过短时独立观测确认，同一组图来自同一跟踪段。</p>
<p>整帧保持输入像素，裁剪最多作整 90 度旋转。未作锐化或合成。这里记录的是位置图像收集结果。</p>
<p><a href="result.json">结果与比对输入接口</a> · <a href="frames.jsonl">逐帧诊断</a></p>
<details><summary>素材来源</summary><pre>{html.escape(json.dumps(report.get('source_attribution', {}),ensure_ascii=False,indent=2))}</pre></details></main></html>'''
    (output / 'review.html').write_text(body, encoding='utf8')
