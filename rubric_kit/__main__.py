"""Allow running rubric-kit as ``python -m rubric_kit``."""

import sys

from rubric_kit.main import main


sys.exit(main())
