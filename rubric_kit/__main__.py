"""Allow running rubric-kit as ``python -m rubric_kit``."""

import sys

from rubric_kit.cli.commands import main


sys.exit(main())
