# Project Structure

This document explains the directory structure of the Random Allocation project and how to handle generated files.

## Directory Structure

```
random_allocation/              # Main project directory
├── random_allocation/          # Source code directory
│   ├── comparisons/            # Code for comparing different schemes
│   ├── examples/               # Example code and experiments
│   ├── other_schemes/          # Implementation of other privacy schemes
│   └── random_allocation_scheme/  # Implementation of random allocation scheme
├── tests/                      # Test files
│   ├── basic_tests.py          # Basic functionality tests
│   ├── check_types.py          # Type checking tests
│   ├── data_handler_tests.py   # Data handler tests
│   └── release_tests.py        # Tests to run before a release
├── docs/                       # Documentation
│   └── PROJECT_STRUCTURE.md    # This document
├── scripts/                    # Utility scripts
│   └── cleanup.sh              # Script to clean up generated files
├── LICENSE                     # Project license
├── pyproject.toml              # Build system configuration and package metadata
├── README.md                   # Main project documentation
├── requirements.txt            # Python package dependencies
└── environment.yml             # Conda environment configuration
```

## Generated Directories (Safe to Remove)

These directories are generated during building, testing, and development. They can be safely removed and will be regenerated as needed:

- `dist/`: Distribution packages generated during build
- `random_allocation.egg-info/`: Package metadata generated during installation
- `.mypy_cache/`: Type checking cache
- `__pycache__/`: Python bytecode cache
- `*.egg-info/`: Egg metadata files

## How to Clean Up Generated Files

Run the cleanup script to remove all generated files:

```bash
./scripts/cleanup.sh
```

## How to Rebuild Artifacts

1. **Building the package**:

```bash
python -m pip install build
python -m build
```

This will regenerate the `dist/` directory with wheel and tarball files.

2. **Installing the package in development mode**:

```bash
pip install -e .
```

This will regenerate the `random_allocation.egg-info/` directory.

3. **Running type checking**:

```bash
mypy random_allocation
```

This will regenerate the `.mypy_cache/` directory. The mypy configuration is stored in `pyproject.toml`.

## Environment Setup

To set up the development environment:

```bash
# Using conda
conda env create -f environment.yml

# Activate the environment
conda activate random_allocation

# Using pip
pip install -r requirements.txt
``` 