"""Module entrypoint: enables ``python -m agent_foundation.resources.tools.task``."""

import sys

from .cli import main

sys.exit(main())
