"""Process-local Windows environment needed by NVIDIA's profile database."""
import os


def ensure_driver_profile_environment():
    """Restore an omitted Windows variable before loading CUDA or NVAPI.

    With ALLUSERSPROFILE missing, driver 566.07 silently reads its factory DRS
    database instead of the saved application profiles. Some launchers sanitize
    this variable. Obtain the real common-data directory from Windows; preserve
    existing values and never edit the machine/user environment or registry.
    """
    if os.name != 'nt' or os.environ.get('ALLUSERSPROFILE'):
        return False
    import ctypes as c
    path = c.create_unicode_buffer(32768)
    get_folder = c.WinDLL('shell32').SHGetFolderPathW
    get_folder.argtypes = [c.c_void_p, c.c_int, c.c_void_p, c.c_uint32, c.c_wchar_p]
    get_folder.restype = c.c_long
    status = get_folder(None, 0x23, None, 0, path)  # CSIDL_COMMON_APPDATA
    if status != 0 or not path.value:
        raise OSError(f'Cannot resolve Windows common application data: HRESULT {status:#x}')
    os.environ['ALLUSERSPROFILE'] = path.value
    return True
