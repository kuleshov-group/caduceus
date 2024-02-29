"""Caduceus"""

import warnings

try:
    from caduceus._version import __version__
except ImportError:
    __version__ = "not-installed"
    warnings.warn(
        "You are running a non-installed version caduceus."
        "If you are running this from a git repo, please run"
        "`pip install -e .` to install the package."
    )