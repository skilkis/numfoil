# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog], and this project adheres to
[Semantic Versioning]. Currently, the project does not have a public API
signified by major version 0. Therefore, the API is subject to frequent
changes and backwards compatibility is not guaranteed.

## [0.1.0] - 2020-10-20

### Added

- 2D geometric feature package into the `geometry/` directory
- Vectorized high-performance panel methods into the `solver/` directory
- `tests/` folder containing [pytest] unit-tests
- Initial assignment report added to the `docs/` directory
- Assignment I task scripts to the `assignment/` directory
- UML activity and class diagram to the `assignment/` directory
- GitHub Actions to automate deployment as well as code-coverage metrics
- Template to permit use of the `src/` Python package layout
- `setup.py` and `setup.cfg` file for setting project metadata, dependencies,
  code formatting, testing, and linting/style options
- `.gitignore` file to specify untracked files
- `.editorconfig` file to enforce line endings and file encodings/whitespace
- `pyproject.toml` file for setting [black] character limit
- Ability to install `gammapy` installable through [pip]
- Apache 2.0 license header and license file
- Sample `test_airfoil` unit-test to trigger `coverage.py` to run
- [README.md] to instruct users how to install `gammapy`
- [CONTRIBUTING.md] file to welcome new developers to the project
- [CHANGELOG.md] to log the [Semantic Versioning] of the project

### Changed

- Location of Python files into the new `src/gammapy/` directory
- Location of old panel methods and moved them into `src/gammapy/legacy/`

### Removed

- Sample test `test_numpy.py`

<!-- Un-wrapped Text Below for References, Links, Images, etc. -->
[Keep a Changelog]: https://keepachangelog.com/en/1.0.0/
[Semantic Versioning]: https://semver.org/spec/v2.0.0.html
[black]: https://black.readthedocs.io/en/stable/
[CONTRIBUTING.md]: CONTRIBUTING.md
[CHANGELOG.md]: CHANGELOG.md
[README.md]: README.md
[pip]: https://pypi.org/project/pip/
