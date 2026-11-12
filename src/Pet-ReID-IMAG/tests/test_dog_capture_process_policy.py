import pytest

from dog_capture.engine import process_policy as policy


@pytest.fixture
def windows(monkeypatch):
    state = {'version': 1, 'control_mask': 4, 'state_mask': 4}
    writes = []
    def system(value=None):
        if value is not None:
            state.update(value)
            writes.append(dict(value))
        return dict(state)
    monkeypatch.setattr(policy, '_windows', lambda: True)
    monkeypatch.setattr(policy, '_windows_policy', system)
    return state, writes


def test_high_qos_preserves_other_bits_and_restores_on_failure(windows):
    state, writes = windows
    with pytest.raises(ValueError, match='capture failed'):
        with policy.cpu_qos('high') as info:
            assert info['applied'] and info['effective'] == 'high'
            assert state == {'version': 1, 'control_mask': 5, 'state_mask': 4}
            raise ValueError('capture failed')
    assert state == {'version': 1, 'control_mask': 4, 'state_mask': 4}
    assert len(writes) == 2


@pytest.mark.parametrize('mode,realtime,applied', [('auto', True, True), ('auto', False, False),
                                                 ('inherit', True, False)])
def test_automatic_policy_only_covers_realtime_entry_points(windows, mode, realtime, applied):
    with policy.cpu_qos(mode, realtime=realtime) as info:
        assert info['applied'] == applied


def test_auto_reports_failure_and_explicit_high_raises(monkeypatch):
    monkeypatch.setattr(policy, '_windows', lambda: True)
    def failed(value=None):
        raise OSError('policy unavailable')
    monkeypatch.setattr(policy, '_windows_policy', failed)
    with policy.cpu_qos('auto', realtime=True) as info:
        assert not info['applied'] and 'policy unavailable' in info['error']
    with pytest.raises(OSError, match='policy unavailable'):
        with policy.cpu_qos('high'):
            pass


def test_auto_is_portable_and_explicit_high_requires_windows(monkeypatch):
    monkeypatch.setattr(policy, '_windows', lambda: False)
    with policy.cpu_qos('auto', realtime=True) as info:
        assert not info['applied']
    with pytest.raises(RuntimeError, match='requires Windows'):
        with policy.cpu_qos('high'):
            pass


def test_nested_scope_restores_the_host_policy(windows):
    state, writes = windows
    with policy.cpu_qos('high'):
        with policy.cpu_qos('high') as info:
            assert info['previous']['control_mask'] == 5
        assert state['control_mask'] == 5
    assert len(writes) == 2 and state['control_mask'] == 4
