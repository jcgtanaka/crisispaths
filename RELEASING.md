# Releasing

This is a checklist. Nothing here is run automatically.

1. Working tree clean on `main`; CI green (`ruff check .`, `pytest`).
2. Run the network smoke test locally: `pytest -m network`.
3. Version: `pyproject.toml` `version` and `src/crisispaths/__init__.py`
   `__version__` must agree. Date the CHANGELOG section of the release.
4. Build in a clean environment: `python -m pip install build twine`, then
   `python -m build` (sdist and wheel in `dist/`).
5. Check the artifacts: `python -m twine check dist/*`; install the wheel in a
   fresh virtual environment and run `crisispaths scenarios` and
   `python -c "import crisispaths"`.
6. Tag: `git tag -a vX.Y.Z -m "crisispaths X.Y.Z"`, then push the tag.
7. Upload (maintainer only): `python -m twine upload dist/*`, after a TestPyPI
   dry run, using a trusted publisher or an API token.
8. Create the GitHub release from the tag with the CHANGELOG section.
