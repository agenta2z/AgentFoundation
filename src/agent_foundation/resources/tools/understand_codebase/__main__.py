"""Module entrypoint: enables ``python -m agent_foundation.resources.tools.understand_codebase``."""
import sys
from .cli import main

sys.exit(main())
