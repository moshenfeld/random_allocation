# Project Structure

This document explains the directory structure of the Random Allocation project and how to handle generated files.

## Directory Structure

```
random_allocation/                    # Main project directory
├── random_allocation/                # Source code directory
│   ├── comparisons/                  # Code for comparing different schemes
│   ├── examples/                     # Example code and experiments
│   │   ├── data/                     # Generated experimental data
│   │   └── plots/                    # Generated experimental plots
│   ├── other_schemes/                # Implementation of other privacy schemes
│   └── random_allocation_scheme/     # Implementation of random allocation scheme
├── tests/                            # Comprehensive test suite
│   ├── test_privacy_params.py        # Parameter validation tests (22 tests)
│   ├── test_mathematical_properties.py  # Mathematical correctness (27 tests)
│   ├── test_edge_cases.py            # Robustness testing
│   ├── test_scheme_comparison.py     # Privacy method comparisons
│   ├── test_integration.py           # Workflow and example validation
│   ├── test_performance.py           # Benchmarking and regression detection
│   ├── test_paper_experiments.py     # Bit-exact experiment validation
│   ├── conftest.py                   # Pytest configuration and fixtures
│   ├── run_test_suite.py             # Test runner with phase-based execution
│   ├── test_requirements.txt         # Test-specific dependencies
│   ├── basic_tests.py                # Legacy basic functionality tests
│   ├── check_types.py                # Type checking tests
│   ├── data_handler_tests.py         # Legacy data handler tests
│   ├── release_tests.py              # Tests to run before a release
│   ├── cleanup.sh                    # Script to clean up generated files
│   └── README.md                     # Test suite documentation
├── docs/                             # Documentation
│   ├── PROJECT_STRUCTURE.md          # This document
│   ├── test_documentation.md         # Comprehensive test suite documentation
│   └── type_annotations_guide.md     # Type annotation guidelines
├── LICENSE                           # Project license
├── pyproject.toml                    # Build system configuration and package metadata
├── README.md                         # Main project documentation
├── requirements.txt                  # Python package dependencies
└── environment.yml                   # Conda environment configuration
```

## Test Suite Organization

The project includes a comprehensive test suite with **110+ tests** organized into categories:

- **Unit Tests**: Core functionality and parameter validation
- **Mathematical Tests**: Privacy property validation and numerical correctness
- **Integration Tests**: End-to-end workflow validation
- **Performance Tests**: Benchmarking and regression detection
- **Reproducibility Tests**: Bit-exact validation of research experiments

**Total Test Runtime**: ~125 seconds for complete suite

## Generated Directories (Safe to Remove)

These directories are generated during building, testing, and development. They can be safely removed and will be regenerated as needed:

- `dist/`: Distribution packages generated during build
- `random_allocation.egg-info/`: Package metadata generated during installation
- `.mypy_cache/`: Type checking cache
- `__pycache__/`: Python bytecode cache directories
- `.pytest_cache/`: Pytest cache
- `.benchmarks/`: Performance benchmarking cache
- `random_allocation/examples/data/`: Generated experimental data
- `random_allocation/examples/plots/`: Generated experimental plots

## How to Clean Up Generated Files

Run the cleanup script to remove all generated files:

```bash
./tests/cleanup.sh
```

## How to Run Tests

### Complete Test Suite
```bash
# Activate environment
conda activate random_allocation

# Run all tests
python tests/run_test_suite.py

# Run with coverage
python tests/run_test_suite.py --coverage
```

### Individual Test Categories
```bash
# Unit tests (most important)
pytest tests/unit/ -v

# Integration tests  
pytest tests/integration/ -v

# Performance tests
pytest tests/performance/ -v

# Research reproducibility tests
pytest tests/paper_experiments/ -v
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
# Using conda (recommended)
conda env create -f environment.yml
conda activate random_allocation

# Using pip
pip install -r requirements.txt
```

## Documentation

- **`README.md`**: Main project documentation with usage examples
- **`docs/test_documentation.md`**: Comprehensive test suite documentation
- **`docs/type_annotations_guide.md`**: Type annotation guidelines
- **`tests/README.md`**: Test-specific setup and usage instructions

## Research Reproducibility

The project includes bit-exact validation of all paper experiments to ensure research reproducibility. See `docs/test_documentation.md` for details on the reproducibility testing framework. 