# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog], and this project adheres to
[Semantic Versioning]. Currently, the project does not have a public API
signified by major version 0. Therefore, the API is subject to frequent
changes and backwards compatibility is not guaranteed.

## Unreleased
<!-- First release will be 0.1.0 as advised by Semantic Versioning -->
<!-- ## [0.1.0] - 2020-04-29 -->

### Added

- Template to permit use of the `src/` Python package layout
- `setup.py` and `setup.cfg` file for setting project metadata, dependencies,
  code formatting, testing, and linting/style options
- `.gitignore` file to specify un-tracked files
- `.editorconfig` file to enforce line endings and file encodings/whitespace
- `pyproject.toml` file for setting [black] character limit
- Ability to install `gammapy` installable through [pip]
- `tests/` folder with a sample test for verifying [pytest]
- Apache 2.0 license header and license file
- [README.md] to instruct users how to install `gammapy`
- [CONTRIBUTING.md] file to welcome new developers to the project
- [CHANGELOG.md] to log the [Semantic Versioning] of the project

### Changed

- Location of Python files into the new `src/gammapy/` directory

<!-- Un-wrapped Text Below for References, Links, Images, etc. -->
[Keep a Changelog]: https://keepachangelog.com/en/1.0.0/
[Semantic Versioning]: https://semver.org/spec/v2.0.0.html
[black]: https://black.readthedocs.io/en/stable/
[CONTRIBUTING.md]: CONTRIBUTING.md
[CHANGELOG.md]: CHANGELOG.md
[README.md]: README.md
[pip]: https://pypi.org/project/pip/
