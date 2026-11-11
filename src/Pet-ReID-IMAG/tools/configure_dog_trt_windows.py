"""Optional NVIDIA power preference scoped to the dedicated tool executable.

Uses the documented NVAPI DRS interface. Does not edit global settings, clocks,
power limits, or the profile of a generic python.exe. Driver writes can require
an administrator. Explicit --enable/--disable is required for mutations.
ABI reference: https://github.com/NVIDIA/nvapi (MIT licensed).
"""
import argparse
import ctypes as c
import hashlib
import json
import os
from pathlib import Path
import sys

U, P, W = c.c_uint32, c.c_void_p, c.c_wchar * 2048


class Profile(c.Structure):
    _fields_ = [('version', U), ('name', W), ('gpu', U), ('predefined', U), ('apps', U), ('settings', U)]


class Application(c.Structure):
    _fields_ = [('version', U), ('predefined', U), ('name', W), ('friendly', W), ('launcher', W)]


class Value(c.Union):
    _fields_ = [('dword', U), ('buffer', c.c_byte * 4100)]


class Setting(c.Structure):
    _pack_ = 4
    _fields_ = [('version', U), ('name', W), ('id', U), ('type', U), ('location', U),
                ('current_predefined', U), ('predefined_valid', U), ('predefined', Value), ('current', Value)]


def structure(cls):
    item = cls()
    item.version = c.sizeof(cls) | (1 << 16)
    return item


def configure(application, action):
    if os.name != 'nt':
        raise RuntimeError('NVIDIA application profiles require Windows')
    application = Path(application).resolve(strict=True)
    if application.name.lower() != 'dog_capture_trt.exe':
        raise ValueError('Only the dedicated dog_capture_trt.exe may be configured')
    dll = c.CDLL('nvapi64.dll')
    dll.nvapi_QueryInterface.argtypes, dll.nvapi_QueryInterface.restype = [U], P
    def fn(identifier, *types):
        pointer = dll.nvapi_QueryInterface(identifier)
        if not pointer:
            raise RuntimeError(f'NVAPI interface unavailable: {identifier:#x}')
        return c.CFUNCTYPE(c.c_int, *types)(pointer)
    error_message = fn(0x6c2d048c, c.c_int, c.c_char_p)
    def check(status):
        if status:
            buffer = c.create_string_buffer(64)
            error_message(status, buffer)
            raise RuntimeError(f'NVAPI {status}: {buffer.value.decode()}; driver profile writes may require Windows administrator rights')
    initialize, unload = fn(0x0150e828), fn(0xd22bdd7e)
    create, destroy = fn(0x0694d52e, c.POINTER(P)), fn(0xdad9cff8, P)
    load, save = fn(0x375dbd6b, P), fn(0xfcbc7e14, P)
    find_app = fn(0xeee566b2, P, c.c_wchar_p, c.POINTER(P), c.POINTER(Application))
    get_profile = fn(0x61cd6fd6, P, P, c.POINTER(Profile))
    make_profile = fn(0xcc176068, P, c.POINTER(Profile), c.POINTER(P))
    make_app = fn(0x4347a9de, P, P, c.POINTER(Application))
    set_setting, get_setting = fn(0x577dd202, P, P, c.POINTER(Setting)), fn(0x73bf8338, P, P, U, c.POINTER(Setting))
    delete_profile = fn(0x17093206, P, P)
    profile_name = 'DogCapture TensorRT ' + hashlib.sha256(str(application).lower().encode()).hexdigest()[:12]
    session, profile = P(), P()
    check(initialize())
    try:
        check(create(c.byref(session)))
        check(load(session))
        app = structure(Application)
        status = find_app(session, str(application), c.byref(profile), c.byref(app))
        if status not in (0, -166):  # NVAPI_EXECUTABLE_NOT_FOUND
            check(status)
        exists = status == 0
        info, setting = structure(Profile), structure(Setting)
        value = None
        if exists:
            check(get_profile(session, profile, c.byref(info)))
            if action != 'status' and info.name != profile_name:
                raise RuntimeError('This application belongs to another profile; no settings changed')
            check(get_setting(session, profile, 0x1057EB71, c.byref(setting)))
            value = setting.current.dword
        if action == 'enable':
            if not exists:
                info.name = profile_name
                check(make_profile(session, c.byref(info), c.byref(profile)))
                app = structure(Application)
                app.name, app.friendly = str(application), 'Dog target-view capture'
                check(make_app(session, profile, c.byref(app)))
            setting = structure(Setting)
            setting.id, setting.current.dword = 0x1057EB71, 1  # PREFERRED_PSTATE_PREFER_MAX
            check(set_setting(session, profile, c.byref(setting)))
            check(save(session))
            exists, value = True, 1
        elif action == 'disable' and exists:
            if info.apps != 1 or info.settings != 1 or value != 1:
                raise RuntimeError('Profile has additional/custom settings; refusing to remove it')
            check(delete_profile(session, profile))
            check(save(session))
            exists, value = False, None
        return {'application': str(application), 'action': action, 'profile_exists': exists,
                'profile_name': info.name if exists else None, 'preferred_pstate': value,
                'global_settings_changed': False}
    finally:
        if session:
            destroy(session)
        unload()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--application', default=str(Path(sys.executable).with_name('dog_capture_trt.exe')))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--enable', action='store_true')
    mode.add_argument('--disable', action='store_true')
    args = parser.parse_args()
    print(json.dumps(configure(args.application, 'enable' if args.enable else 'disable' if args.disable else 'status'), indent=2))


if __name__ == '__main__':
    main()
