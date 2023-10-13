try:
    from ._version_generated import __version__
except ImportError:
    __version__ = "unreleased"

import pyhd.processing as processing