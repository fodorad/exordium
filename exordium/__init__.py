from importlib.metadata import version
from pathlib import Path

try:
    __version__ = version("exordium")
except Exception:
    __version__ = "unknown"

# Module level constants
PROJECT_ROOT = Path(__file__).parents[1]
"""Root directory of the repository (parent of the ``exordium`` package)."""
PACKAGE_ROOT = PROJECT_ROOT / "exordium"
"""Root directory of the ``exordium`` package source tree."""
RESOURCE_DIR = PROJECT_ROOT / "resources"
"""Path to bundled resource files shipped with the package."""
TEST_DIR = PROJECT_ROOT / "tests"
"""Path to the test suite directory."""
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures"
"""Path to test fixture files — sample images, audio clips, and short videos."""
WEIGHT_DIR = Path().home() / ".cache" / "torch" / "hub" / "checkpoints"
"""Default cache directory for downloaded model weights (``~/.cache/torch/hub/checkpoints``)."""
