"""Sources, worker, outputs and the CLI on synthetic media (no model files)."""
import json
import time

import cv2
import numpy as np
import pytest

from dog_capture.engine import CaptureEngine
from dog_capture.engine.cli import main
from dog_capture.engine.io import (DetectionLog, ImageSource, LatestFrameRunner, VideoSource, choose_output_dir,
                                   imwrite)
from dog_engine_synthetic import render


def write_video(path, yaws, fps=15):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), fps, (640, 480))
    assert writer.isOpened()
    lines = []
    for i, yaw in enumerate(yaws):
        image, dets = render(size=(480, 640), center=(320 + i, 240), iod=90, yaw=float(yaw))
        writer.write(image)
        lines.append(json.dumps({'frame_index': i, 'detections': [dets[0].to_dict()]}))
    writer.release()
    return lines


@pytest.fixture
def media(tmp_path):
    yaws = [0] * 12 + [-.5] * 12 + [.45] * 12
    video = tmp_path / 'dog.avi'
    lines = write_video(video, yaws)
    log = tmp_path / 'dets.jsonl'
    log.write_text('\n'.join(lines) + '\n')
    return video, log


def test_video_timestamps_are_strictly_increasing(media):
    source = VideoSource(media[0])
    times = [t for _, t, _ in source]
    source.close()
    assert len(times) == 36 and all(b > a for a, b in zip(times, times[1:]))
    assert times[1] == pytest.approx(1 / 15, abs=1e-3)


def test_image_source_skips_unreadable_files(tmp_path):
    imwrite(tmp_path / 'a.png', render()[0])
    (tmp_path / 'b.jpg').write_bytes(b'not an image')
    source = ImageSource(tmp_path)
    assert len(list(source)) == 1 and source.describe()['unreadable']


def test_output_directory_is_never_overwritten(tmp_path):
    first = choose_output_dir(tmp_path / 'out', 'x.mp4', tmp_path)
    (first / 'result.json').write_text('{}')
    second = choose_output_dir(tmp_path / 'out', 'x.mp4', tmp_path)
    assert second != first and second.parent == first and not any(second.iterdir())


def test_detection_log_accepts_v0_observations(tmp_path):
    path = tmp_path / 'legacy.jsonl'
    path.write_text('\n'.join([
        json.dumps({'frame_index': 0, 'observation': {'bbox_xyxy': [0, 0, 10, 10], 'confidence': .9,
                                                      'landmarks_xy': [[1, 1]] * 5}}),
        json.dumps({'frame_index': 2, 'observation': None})]))
    log = DetectionLog(path)
    assert len(log.get(0)) == 1 and log.get(1) == [] and log.get(2) == []


def test_latest_frame_runner_drops_instead_of_lagging_and_reports_errors():
    class Slow:
        def process(self, frame, timestamp, index, detections=None):
            time.sleep(.03)
            return None
    runner = LatestFrameRunner(Slow())
    frame = np.zeros((32, 32, 3), np.uint8)
    for i in range(20):
        runner.submit(frame, i / 100, i)
        time.sleep(.005)
    runner.close()
    assert runner.dropped > 5

    class Broken:
        def process(self, *args, **kwargs):
            raise RuntimeError('boom')
    runner = LatestFrameRunner(Broken())
    runner.submit(frame, 0, 0)
    time.sleep(.2)
    with pytest.raises(RuntimeError, match='worker failed'):
        runner.submit(frame, .1, 1)


def test_cli_run_with_upstream_detections_writes_complete_results(media, tmp_path):
    video, log = media
    out = tmp_path / 'run'
    code = main(['run', str(video), '--detections', str(log), '--output', str(out), '--no-body-gate',
                 '--preview', '--quiet'])
    assert code == 0
    report = json.loads((out / 'result.json').read_text())
    assert report['selection']['coverage_complete'], report['selection']
    for view, items in report['selection']['views'].items():
        for key in ('aligned', 'frame', 'face'):
            assert (out / items[0][key]).is_file()
        aligned = cv2.imread(str(out / items[0]['aligned']))
        assert aligned.shape == (224, 224, 3)
    frame = cv2.imread(str(out / report['selection']['views']['front'][0]['frame']))
    assert frame.shape == (480, 640, 3)
    assert (out / 'review.html').read_text(encoding='utf8').count('<article>') == 3
    assert sum(1 for _ in (out / 'frames.jsonl').open()) == 36
    assert (out / 'preview.mp4').stat().st_size > 0
    assert report['stats']['frames_processed'] == 36


def test_cli_realtime_replay_counts_drops(media, tmp_path):
    video, log = media
    code = main(['run', str(video), '--detections', str(log), '--output', str(tmp_path / 'rt'),
                 '--no-body-gate', '--realtime', '--quiet', '--no-frame-log'])
    report = json.loads((tmp_path / 'rt' / 'result.json').read_text())
    stats = report['stats']
    assert code in (0, 2) and stats['realtime']
    assert stats['frames_processed'] + stats['frames_dropped'] == stats['frames_read']
    assert stats['capture_to_result_ms']['p95'] > 0


def test_cli_config_and_models_commands(tmp_path, capsys):
    assert main(['config', '--write', str(tmp_path / 'c.json')]) == 0
    assert json.loads((tmp_path / 'c.json').read_text())['selection']['mode'] == 'hybrid'
    assert main(['models']) == 0
    assert 'yolox_tiny' in capsys.readouterr().out


def test_cli_until_done_stops_early(media, tmp_path):
    video, log = media
    code = main(['run', str(video), '--detections', str(log), '--output', str(tmp_path / 'u'), '--no-body-gate',
                 '--quiet', '--mode', 'fast', '--until-done'])
    report = json.loads((tmp_path / 'u' / 'result.json').read_text())
    assert code == 0 and report['selection']['coverage_complete']
    assert report['stats']['frames_read'] < 36
