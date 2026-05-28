"""Module entrypoint: enables ``python -m agent_foundation.resources.tools.sop``."""
import sys
from .cli import main

sys.exit(main())
