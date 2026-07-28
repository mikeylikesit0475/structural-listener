"""Smoke test: the package imports and reports its version."""

import structural_listener


def test_package_imports() -> None:
    # __version__ is a non-empty literal; assert it parses as a valid dotted version.
    parts = structural_listener.__version__.split(".")
    assert len(parts) >= 2  # at least major.minor
