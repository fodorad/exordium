import toml
from os import PathLike
from pathlib import Path

# Module level constants
PROJECT_ROOT = Path(__file__).parents[1]
PACKAGE_ROOT = PROJECT_ROOT / 'exordium'
TOOL_ROOT = PROJECT_ROOT / 'tools'
RESOURCE_DIR = PROJECT_ROOT / 'resources'
DATA_DIR = PROJECT_ROOT / 'data'
WEIGHT_DIR = Path().home() / '.cache' / 'torch' / 'hub' / 'checkpoints'

pyproject_data = toml.load(PROJECT_ROOT / 'pyproject.toml')
__version__ = pyproject_data['project']['version']

# Type aliases
PathType = str | PathLike

# Example data
EXAMPLE_VIDEO_PATH = DATA_DIR / 'videos' / 'example_multispeaker.mp4'
EXAMPLE_VDET_CACHE_PATH = DATA_DIR / 'example_multispeaker' / 'cache.vdet'