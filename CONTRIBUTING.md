# Contributing

Thanks for improving Daily Code Lab. Keep examples runnable, prefer the Python
standard library unless an added dependency has a clear benefit, and document any
new command-line options or behavioral assumptions.

## Development checks

1. Use Python 3.10 or newer.
2. Run **python -m unittest discover -s tests -v** from the repository root.
3. Smoke-test the relevant example or CLI command.
4. Add regression tests for bug fixes and edge cases.
5. Keep changes focused and use descriptive commit messages.

CSV Inspector intentionally does not modify source files, upload data, or make
network requests. Do not add telemetry or file transmission without an explicit
project-level design discussion.
