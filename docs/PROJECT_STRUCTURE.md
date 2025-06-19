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
│   ├── basic/                        # Basic functionality tests (27 tests)
│   │   ├── test_basic_01_functionality.py        # Parameter validation, Gaussian mechanism
│   │   ├── test_basic_02_core_allocation_methods.py  # Core allocation methods
│   │   ├── test_basic_03_direct_allocation_add.py    # Direct allocation (add direction)
│   │   ├── test_basic_03_direct_allocation_remove.py # Direct allocation (remove direction)
│   │   └── test_basic_04_direct_RDP_add.py           # RDP-based direct method
│   ├── full/                         # Full comprehensive tests (37 tests)
│   │   ├── test_full_01_additional_allocation_methods.py  # Extended allocation methods
│   │   ├── test_full_02_other_schemes.py                  # Other privacy schemes
│   │   ├── test_full_03_mathematical_properties.py       # Mathematical correctness
│   │   └── test_full_04_type_annotations.py              # Type annotation compliance
│   ├── release/                      # Release validation tests (26 tests)
│   │   ├── test_release_01_comprehensive_coverage.py     # Complex scenarios and edge cases
│   │   └── test_release_02_complete_type_annotations.py  # Complete type validation
│   ├── paper/                        # Research reproducibility tests
│   │   └── test_paper_01_experiments.py                  # Paper experiment reproduction
│   ├── run_tests.py                  # Hierarchical test runner
│   ├── README.md                     # Test suite documentation
│   └── TEST_STRUCTURE.md             # Detailed test organization
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

The project includes a comprehensive test suite with **90+ tests** organized into four hierarchical levels:

- **Basic Tests** (27 tests): Core functionality and parameter validation (~1-5s)
- **Full Tests** (37 tests): Comprehensive validation including mathematical properties (~5-30s)
- **Release Tests** (26 tests): Integration and comprehensive validation (~30+s)
- **Paper Tests** (Variable): Research reproducibility and experiment validation

**Total Test Runtime**: ~1-2 minutes for release-level tests (excluding paper experiments)

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

Remove generated directories manually or use standard Python cleanup:

```bash
# Remove Python cache files
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete

# Remove build artifacts
rm -rf dist/ random_allocation.egg-info/ .mypy_cache/ .pytest_cache/

# Remove generated test data (optional)
rm -rf random_allocation/examples/data/
rm -rf random_allocation/examples/plots/
```

## How to Run Tests

### Hierarchical Test Execution
```bash
# Activate environment
conda activate random_allocation

# Run basic tests (fast development feedback)
python tests/run_tests.py basic

# Run full validation (development and pre-release)
python tests/run_tests.py full

# Run complete validation (release preparation)
python tests/run_tests.py release

# Run all tests including paper experiments
python tests/run_tests.py paper

# Additional options
python tests/run_tests.py full --fast    # Skip slow tests
python tests/run_tests.py basic -x       # Stop on first failure
```

### Individual Test Categories
```bash
# Basic functionality tests
pytest tests/basic/ -v

# Full comprehensive tests  
pytest tests/full/ -v

# Release validation tests
pytest tests/release/ -v

# Research reproducibility tests
pytest tests/paper/ -v
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
- **`tests/TEST_STRUCTURE.md`**: Detailed test organization and structure

## Research Reproducibility

The project includes comprehensive validation of research experiments to ensure reproducibility. The paper tests validate bit-exact reproduction of research results. See `tests/TEST_STRUCTURE.md` for details on the test organization and `tests/paper/` for research-specific validation. 