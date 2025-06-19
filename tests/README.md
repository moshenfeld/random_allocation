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

The test suite is organized into four hierarchical levels:

- **Basic** (`basic/`): Core functionality, fast execution (~1-5s) - 27 tests
- **Full** (`full/`): Comprehensive testing (~5-30s) - 37 tests
- **Release** (`release/`): Integration and comprehensive validation (~30+s) - 26 tests  
- **Paper** (`paper/`): Research reproducibility (variable timing) - Research-specific tests

**Total**: 90+ tests across all levels

## Documentation

- **🏗️ Test Organization**: [`TEST_STRUCTURE.md`](TEST_STRUCTURE.md) - Detailed hierarchy and structure
- **📋 Project Structure**: [`../docs/PROJECT_STRUCTURE.md`](../docs/PROJECT_STRUCTURE.md) - Overall project organization
- **📖 Main Documentation**: [`../README.md`](../README.md) - Package usage and examples

## Test File Structure

### Basic Tests (`basic/`)
- `test_basic_01_functionality.py` - Parameter validation, Gaussian mechanism
- `test_basic_02_core_allocation_methods.py` - Core allocation methods  
- `test_basic_03_direct_allocation_add.py` - Direct allocation (add direction)
- `test_basic_03_direct_allocation_remove.py` - Direct allocation (remove direction)
- `test_basic_04_direct_RDP_add.py` - RDP-based direct method

### Full Tests (`full/`)
- `test_full_01_additional_allocation_methods.py` - Extended allocation methods
- `test_full_02_other_schemes.py` - Other privacy schemes (local, Poisson, shuffle)
- `test_full_03_mathematical_properties.py` - Mathematical correctness
- `test_full_04_type_annotations.py` - Type annotation compliance

### Release Tests (`release/`)
- `test_release_01_comprehensive_coverage.py` - Complex scenarios and edge cases
- `test_release_02_complete_type_annotations.py` - Complete type validation

### Paper Tests (`paper/`)
- `test_paper_01_experiments.py` - Research experiment reproduction

## Current Status

✅ **All core tests passing** - The test suite successfully validates all privacy allocation methods with comprehensive coverage of mathematical properties, edge cases, and type annotations.

### Recent Updates
- Fixed decomposition method bug (NameError resolved)
- Reorganized release test files (renamed for consistency)
- Verified no inappropriate error suppression across test suite
- Added comprehensive type annotation validation

## Advanced Usage

```bash
# Run with additional options
python run_tests.py full --fast        # Skip slow tests
python run_tests.py basic -x           # Stop on first failure  
python run_tests.py release --verbose  # Detailed output

# Run specific test categories
pytest basic/ -v                       # Only basic tests
pytest full/test_full_03_mathematical_properties.py -v  # Specific file
pytest -k "epsilon" -v                 # Tests matching pattern
``` 