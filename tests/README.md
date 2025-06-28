# Random Allocation Test Suite

This directory contains comprehensive tests for the random_allocation package.

## Quick Start

```bash
# Run all tests except paper experiments (recommended)
python run_tests.py release

# Run basic tests only (fast feedback during development)
python run_tests.py basic

# Run basic + full tests (thorough development testing)  
python run_tests.py full

# Run all tests including paper experiments (research validation)
python run_tests.py paper

# Run with pytest directly (basic usage)
pytest -v
```

## Test Organization

The test suite is organized into four hierarchical levels focused on different aspects:

- **Basic** (`basic/`): Core functionality, fast execution (~1-5s) - 10 tests
- **Full** (`full/`): Extended schemes and cross-method validation (~5-30s) - 28 tests  
- **Release** (`release/`): Comprehensive validation (~30+s) - 872 tests
- **Paper** (`paper/`): Research reproducibility (variable timing) - Research-specific tests

**Total**: 924 tests across all levels, with 476 comprehensive edge case tests and 370 monotonicity tests

## Documentation

- **🏗️ Test Organization**: [`TEST_STRUCTURE.md`](TEST_STRUCTURE.md) - Detailed hierarchy and structure
- **📋 Project Structure**: [`../docs/PROJECT_STRUCTURE.md`](../docs/PROJECT_STRUCTURE.md) - Overall project organization
- **📖 Main Documentation**: [`../README.md`](../README.md) - Package usage and examples

## Test File Structure

### Basic Tests (`basic/`)
- `test_basic_01_functionality.py` - Parameter validation, Gaussian mechanism, core objects

### Full Tests (`full/`)
- `test_full_01_additional_allocation_methods.py` - Extended allocation methods
- `test_full_02_other_schemes.py` - Other privacy schemes (local, Poisson, shuffle)
- `test_full_03_utility_functions.py` - Core utility functions (search, bounds, convergence)

### Release Tests (`release/`)
- `test_release_01_complete_type_annotations.py` - Complete type validation (26 tests)
- `test_release_02_monotonicity.py` - **Comprehensive monotonicity tests (370 tests)**
- `test_release_03_edge_cases.py` - **Comprehensive edge case tests (476 tests)**

### Paper Tests (`paper/`)
- `test_paper_01_experiments.py` - Research experiment reproduction

## Current Status

✅ **All core tests passing** - The modernized test suite successfully validates all privacy allocation methods with mathematical precision:

- **476 Edge Case Tests**: Mathematically valid boundary condition testing
- **370 Monotonicity Tests**: Mathematical property validation  
- **26 Type Annotation Tests**: Complete type coverage
- **Cross-Scheme Validation**: All privacy schemes tested and compared

### Recent Modernization (2025)
- **Eliminated Invalid Tests**: Removed 190 mathematically invalid test combinations
- **Mathematical Precision**: Only valid epsilon-delta relationships tested
- **Reduced Unnecessary Skips**: 60% reduction in meaningless skipped tests
- **Function Existence Validation**: Only existing functions are tested

### Key Test Features
- **Valid Edge Cases Only**: Edge cases generate tests only for mathematically compatible parameters
- **Monotonicity Validation**: All methods tested for mathematical correctness  
- **Fast Unit Tests**: Core functionality validated quickly for development
- **Research Validation**: End-to-end experiment reproducibility

## Advanced Usage

```bash
# Run with additional options
python run_tests.py full --fast        # Skip slow tests
python run_tests.py basic -x           # Stop on first failure  
python run_tests.py release --verbose  # Detailed output

# Run specific test categories
pytest basic/ -v                       # Only basic tests
pytest release/test_release_03_edge_cases.py -v  # Specific file
pytest -k "epsilon" -v                 # Tests matching pattern

# Individual suite runners
python run_basic_suite.py             # Basic tests only
python run_full_suite.py              # Full tests only
python run_release_suite.py           # Release tests only
```

## Skip Categories

The modernized test suite has three legitimate skip categories:

1. **Invalid Edge Case** (57.6%): Edge cases mathematically incompatible with specific schemes
2. **Documented Bug** (23.2%): Known issues in specific implementations  
3. **Computational Timeout** (19.2%): Edge cases causing algorithmic complexity timeouts

All skips represent genuine mathematical or computational limitations, not parametrization errors.