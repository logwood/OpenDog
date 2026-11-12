"""Process-scoped CPU QoS for latency-sensitive native capture entry points.

Windows can infer a lower QoS for background work on hybrid CPUs. HighQoS is
an explicit execution-speed hint, not a priority-class or affinity change.
The original policy is restored on exit. Library build_engine() never sets
process policy: a host must opt into this scope itself, once around its run.

https://learn.microsoft.com/windows/win32/api/processthreadsapi/nf-processthreadsapi-setprocessinformation
"""
from contextlib import contextmanager
import sys


def _windows():
    return sys.platform == 'win32'


def _windows_policy(value=None):
    import ctypes as ct
    class State(ct.Structure):
        _fields_ = [('Version', ct.c_ulong), ('ControlMask', ct.c_ulong), ('StateMask', ct.c_ulong)]
    kernel = ct.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ct.c_void_p
    function = kernel.GetProcessInformation if value is None else kernel.SetProcessInformation
    function.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_ulong]
    function.restype = ct.c_int
    state = State(1, 0, 0) if value is None else State(value['version'], value['control_mask'], value['state_mask'])
    if not function(kernel.GetCurrentProcess(), 4, ct.byref(state), ct.sizeof(state)):
        raise ct.WinError(ct.get_last_error())
    return {'version': state.Version, 'control_mask': state.ControlMask, 'state_mask': state.StateMask}


@contextmanager
def cpu_qos(mode='auto', *, realtime=False):
    """auto: HighQoS for Windows real-time capture; inherit: leave host policy.

Explicit high requests fail visibly if unavailable. Automatic requests fall
back with an error recorded in the returned metadata. Scope is process-wide;
use at an entry point, not independently from concurrent capture workers.
"""
    if mode not in ('auto', 'inherit', 'high'):
        raise ValueError('cpu_qos must be auto, inherit or high')
    wanted = mode == 'high' or (mode == 'auto' and realtime and _windows())
    info = {'requested': mode, 'effective': 'inherited', 'applied': False, 'platform': sys.platform}
    previous, changed = None, False
    if wanted:
        if not _windows():
            raise RuntimeError('Explicit HighQoS requires Windows')
        try:
            previous = _windows_policy()
            target = {**previous, 'control_mask': previous['control_mask'] | 1,
                      'state_mask': previous['state_mask'] & ~1}
            if target != previous:
                _windows_policy(target)
                changed = True
            active = _windows_policy()
            if not active['control_mask'] & 1 or active['state_mask'] & 1:
                raise OSError('Windows did not activate the requested HighQoS policy')
            info.update(effective='high', applied=True, previous=previous, active=active)
        except (OSError, AttributeError) as error:
            if changed:
                _windows_policy(previous)
                changed = False
            if mode == 'high':
                raise
            info['error'] = str(error)
    try:
        yield info
    finally:
        if changed:
            _windows_policy(previous)
