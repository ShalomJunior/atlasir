"""Support ``python -m atlasir`` with the same interface as the installed CLI."""

from atlasir.cli import main

raise SystemExit(main())
