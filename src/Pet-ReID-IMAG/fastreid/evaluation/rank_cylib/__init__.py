# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""


def compile_helper():
    """Compile helper function at runtime. Make sure this
    is invoked on a single process.

    Cython is an optional acceleration path.  A missing compiler dependency
    must leave the pure-Python rank implementation available instead of
    aborting an otherwise valid training run during final evaluation.
    """
    import os
    import subprocess

    path = os.path.abspath(os.path.dirname(__file__))
    ret = subprocess.run(["make", "-C", path])
    if ret.returncode != 0:
        print("Making cython reid evaluation module failed; using Python rank evaluation.")
        return False
    return True
