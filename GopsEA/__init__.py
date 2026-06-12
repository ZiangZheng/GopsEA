from pathlib import Path


_INNER_PACKAGE = Path(__file__).with_name("GopsEA")
__path__ = [str(_INNER_PACKAGE)]

from .utils.configclass import configclass  # noqa: E402,F401
