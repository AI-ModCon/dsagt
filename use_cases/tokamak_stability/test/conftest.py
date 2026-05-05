import sys
from pathlib import Path

# Make the use-case directory importable so tests can do
# "from hdf5 import ..." and "from m3dc1_tools import ..."
sys.path.insert(0, str(Path(__file__).parent.parent))
