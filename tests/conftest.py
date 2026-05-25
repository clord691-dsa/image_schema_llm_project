from __future__ import annotations

import sys
from pathlib import Path

# Allow `pytest` from the repository root before an editable install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
