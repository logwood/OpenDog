"""Data leakage, deployment crop geometry and masked-supervision regressions."""
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from import_dog_landmark_weights import module_path
from prepare_dog_video_distillation import read_manifest, sha256
from train_dog_video_landmarks import VideoCrops, load_rows, loss_function, validation_guard, validation_score, yaw_numpy
from dog_capture.engine.refine import apply, crop_matrix, warp_crop


def source(identifier,split,group=None,digest=None):
    return {'id':identifier,'split':split,'dog_group':group or identifier,
            'sha256':digest or identifier,'path':identifier+'.mp4'}


def manifest(tmp_path,rows,**extra):
    path=tmp_path/'sources.json'
    path.write_text(json.dumps({'sources':rows,**extra}),encoding='utf8')
    return path


def test_split_is_by_dog_not_frame(tmp_path):
    p=manifest(tmp_path,[source('a','train','same_dog'),source('b','val','same_dog')])
    with pytest.raises(ValueError,match='leakage'):
        read_manifest(p,tmp_path,verify=False)


def test_duplicate_source_under_different_name_is_rejected(tmp_path):
    p=manifest(tmp_path,[source('a','train',digest='same'),source('b','val',digest='same')])
    with pytest.raises(ValueError,match='Duplicate video content'):
        read_manifest(p,tmp_path,verify=False)


def test_protected_test_cannot_be_training_or_validation(tmp_path):
    p=manifest(tmp_path,[source('a','train'),source('b','val')],protected_sha256=['b'])
    with pytest.raises(ValueError,match='Protected regression'):
        read_manifest(p,tmp_path,verify=False)


def test_source_hash_checked_before_inference(tmp_path):
    (tmp_path/'a.mp4').write_bytes(b'changed video')
    p=manifest(tmp_path,[source('a','train'),source('b','val')])
    with pytest.raises(ValueError,match='hash mismatch'):
        read_manifest(p,tmp_path)


def test_export_scope_maps_nested_and_sequential_convs():
    assert module_path('/m/layers.2/m.0/cv1/conv/Conv')=='layers.2.m.0.cv1.conv'
    assert module_path('/m/lat5/lat5.0/Conv')=='lat5.0'
    with pytest.raises(ValueError,match='Unexpected Conv scope'):
        module_path('/different_architecture/Conv')


def example_row(tmp_path):
    yy,xx=np.mgrid[:500,:800]
    frame=np.stack([xx%256,yy%256,(xx+yy)%256],axis=-1).astype(np.uint8)
    cv2.imwrite(str(tmp_path/'frame.png'),frame)
    return {'id':'a','image':'frame.png','image_sha256':sha256(tmp_path/'frame.png'),
            'points':[[340,220],[460,220],[430,280],[420,310],[300,170],[500,170]],
            'box':[250,100,550,400],'upright_angle':0,'point_weights':[1,1,1,1,0,1],
            'yaw_supported':True,'yaw':.2381,'bin':'front','video_id':'dog_a',
            'source_sha256':'a','split':'train','dog_group':'dog_a'}


def test_validation_crop_deterministic_and_matches_runtime(tmp_path):
    row=example_row(tmp_path)
    dataset=VideoCrops(tmp_path,[row],train=False)
    first,second=dataset[0],dataset[0]
    for a,b in zip(first,second):
        np.testing.assert_array_equal(a,b)
    frame=cv2.imread(str(tmp_path/'frame.png'))
    # Recover the affine from three non-collinear labeled coordinates. The
    # image and point paths must describe the same pixel-centre geometry.
    matrix=np.eye(3)
    matrix[:2]=cv2.getAffineTransform(np.float32(row['points'][:3]),first[1][:3]*128)
    expected=warp_crop(frame,matrix,128,antialias=True)
    observed=(first[0].transpose(1,2,0)[:,:,::-1]*255).round().astype(np.uint8)
    np.testing.assert_allclose(observed,expected,atol=1)
    assert first[2][4]==0


def test_yaw_rotation_invariant_and_mirror_changes_sign():
    p=np.float64([[10,10],[30,10],[28,25],[20,35],[5,0],[35,0]])
    initial=yaw_numpy(p)
    rotated=apply(crop_matrix([0,0,40,40],123,.35,128),p)
    assert yaw_numpy(rotated)==pytest.approx(initial)
    flipped=p.copy();flipped[:,0]=100-flipped[:,0];flipped=flipped[[1,0,2,3,5,4]]
    assert yaw_numpy(flipped)==pytest.approx(-initial)


def test_changed_labels_and_split_tampering_rejected(tmp_path):
    row=example_row(tmp_path)
    labels=tmp_path/'labels.jsonl'
    labels.write_text(json.dumps(row)+'\n',encoding='utf8')
    data={'labels_sha256':sha256(labels),'source_manifest':{'sources':[source('dog_a','val',digest='a')]}}
    (tmp_path/'dataset.json').write_text(json.dumps(data),encoding='utf8')
    with pytest.raises(ValueError,match='fixed source manifest'):
        load_rows(tmp_path)
    labels.write_text('{}\n',encoding='utf8')
    with pytest.raises(ValueError,match='labels changed'):
        load_rows(tmp_path)


def test_unsupported_points_receive_no_teacher_gradient():
    torch=pytest.importorskip('torch')
    xy=torch.full((1,8,2),.5,requires_grad=True)
    heat=torch.zeros((1,8,256),requires_grad=True)
    anchor=(xy.detach().clone(),)
    points=torch.full((1,6,2),.4)
    mask=torch.tensor([[1.,0,1,0,0,0]])
    axis=torch.full((1,2,2),.4)
    axis_mask=torch.tensor([[1.,0]])
    loss=loss_function((xy,None,None,heat,None),anchor,[points,mask,axis,axis_mask,torch.zeros(1)])
    loss.backward()
    assert torch.isfinite(loss)
    assert xy.grad[0,0].abs().sum()>0
    assert xy.grad[0,1].abs().sum()==0
    assert heat.grad[0,1].abs().sum()==0
    assert heat.grad[0,7].abs().sum()==0


def test_heat_targets_on_training_device():
    torch=pytest.importorskip('torch')
    from train_dog_face_landmarks import heat_targets
    device='cuda' if torch.cuda.is_available() else 'cpu'
    points=torch.tensor([[[.5,.5],[.4,.6]]],device=device)
    target=heat_targets(points,16)
    assert target.device==points.device
    torch.testing.assert_close(target.sum(-1),torch.ones((1,2),device=device))


def test_checkpoint_selection_equal_video_weight():
    # A prolific easy video may not hide the error on a short hard video.
    metrics={'by_video':{'long':{'nme':{'mean':.1,'samples':1000},'yaw_mae':{'mean':.1}},
                         'short':{'nme':{'mean':.4,'samples':5},'yaw_mae':{'mean':.2}}}}
    assert validation_score(metrics)==pytest.approx(.4)
    with pytest.raises(ValueError,match='No usable validation'):
        validation_score({'by_video':{}})


def test_better_yaw_cannot_hide_regressed_landmarks_or_missing_side_validation():
    baseline={'nme':{'mean':.14},'yaw_mae':{'mean':.075}}
    candidate={'nme':{'mean':.153},'yaw_mae':{'mean':.064},'by_pose':{'front':{}}}
    result=validation_guard(baseline,candidate)
    assert not result['passed'] and not result['automatic_activation']
    assert set(result['reasons'])=={'validation_nme_regressed','no_supported_left_validation','no_supported_right_validation'}


def test_full_video_benchmark_really_runs_the_tracker(tmp_path,monkeypatch):
    from benchmark_dog_capture_v1 import video_benchmark
    from dog_capture.engine import CaptureConfig
    from dog_capture.engine import io
    from dog_engine_synthetic import BlobDetector,BlobLandmarker,render
    frame,_=render(roll=90)
    detector,landmarker=BlobDetector(),BlobLandmarker()
    detector.describe=lambda:{'kind':'synthetic_detector'}
    landmarker.describe=lambda:{'kind':'synthetic_landmarker'}
    class Source:
        def __iter__(self):
            return iter((frame,i/30,i) for i in range(20))
        def describe(self):
            return {'fps':30}
        def close(self):
            pass
    monkeypatch.setattr(io,'VideoSource',lambda _:Source())
    original=landmarker.detect
    config=CaptureConfig().merged({'runtime':{'rotate':'0'}})
    result=video_benchmark(config,detector,None,tmp_path/'unused.mp4',False,tmp_path/'run',1,20,landmarker=landmarker)
    assert result['frames_processed']==20
    assert result['model_calls']['landmarks']>=20
    assert result['landmarks']['kind']=='synthetic_landmarker'
    # RunWriter saves the face upright and transforms its metadata accordingly.
    assert abs(result['selection']['front'][0]['pose']['roll_deg'])<5
    assert landmarker.detect==original  # profiling wrapper cannot leak into another run
