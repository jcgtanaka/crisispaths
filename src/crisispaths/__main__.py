"""``python -m crisispaths`` is the same as the ``crisispaths`` command."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
