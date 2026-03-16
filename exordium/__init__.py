from importlib.metadata import version
from pathlib import Path

try:
    __version__ = version("exordium")
except Exception:
    __version__ = "unknown"

# Module level constants
PROJECT_ROOT = Path(__file__).parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "exordium"
RESOURCE_DIR = PROJECT_ROOT / "resources"
TEST_DIR = PROJECT_ROOT / "tests"
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures"
WEIGHT_DIR = Path().home() / ".cache" / "torch" / "hub" / "checkpoints"
