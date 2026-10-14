# Security policy

## Supported versions

Only the latest released version (and the `main` branch) receives fixes. The
project is alpha software.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private
vulnerability reporting ("Security" tab, "Report a vulnerability") on the
repository. Include the version, the command or code that triggers the problem
and its impact. You should receive a first response within a few days.

## Scope notes

- crisispaths reads CSV, parquet and YAML files you provide and imports the
  strategy module you name with `--strategy module:fn` (it executes that code).
  Do not run it on untrusted strategy modules or untrusted YAML families.
- Results files are plain CSV and JSON; no pickles are written to disk.
