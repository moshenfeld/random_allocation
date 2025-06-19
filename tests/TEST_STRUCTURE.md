# Test Structure and Organization

## Hierarchical Test Structure

The tests are organized in a four-tier hierarchy:

### 🔵 BASIC Tests (`basic/`)
**Purpose**: Core functionality validation
**Runtime**: Fast (~1-5 seconds)
**Run Level**: Always included

- `test_basic_01_functionality.py` - Parameter validation, Gaussian mechanism, direction enum
- `test_basic_02_core_allocation_methods.py` - Core allocation methods (decomposition, analytic)
- `test_basic_03_direct_allocation_add.py` - Direct allocation method (add direction)
- `test_basic_03_direct_allocation_remove.py` - Direct allocation method (remove direction)  
- `test_basic_04_direct_RDP_add.py` - RDP-based direct method (add direction)

### 🟡 FULL Tests (`full/`)  
**Purpose**: Comprehensive feature testing
**Runtime**: Moderate (~5-30 seconds)
**Run Level**: Development and pre-release

- `test_full_01_additional_allocation_methods.py` - Extended allocation methods, direction consistency
- `test_full_02_other_schemes.py` - Non-allocation privacy schemes (local, Poisson, shuffle)
- `test_full_03_mathematical_properties.py` - Mathematical correctness, boundary conditions
- `test_full_04_type_annotations.py` - Type annotation compliance and validation

### 🔴 RELEASE Tests (`release/`)
**Purpose**: Integration and comprehensive validation  
**Runtime**: Slow (~30+ seconds)
**Run Level**: Pre-release and CI/CD

- `test_release_01_comprehensive_coverage.py` - Complex scenarios, edge cases, boundary conditions
- `test_release_02_complete_type_annotations.py` - Complete type annotation coverage and validation

### 📄 PAPER Tests (`paper/`)
**Purpose**: Research reproducibility and paper experiment validation
**Runtime**: Variable (experiment-dependent)
**Run Level**: Research validation and paper submission

- `test_paper_01_experiments.py` - Paper experiment integration and reproducibility tests

## Test Runner Usage

```bash
# Run only basic tests (fast feedback during development)
python run_tests.py basic

# Run basic + full tests (thorough development testing)
python run_tests.py full

# Run basic + full + release tests (complete validation)
python run_tests.py release

# Run all tests including paper experiments
python run_tests.py paper
# or
python run_tests.py all

# Additional options
python run_tests.py full --fast        # Skip slow tests
python run_tests.py basic -x           # Stop on first failure
python run_tests.py release --verbose  # Detailed output
```

## Test Categories by Content

### Mathematical Correctness
- **Level 1 (Basic)**: Parameter validation, core mathematical properties
- **Level 2 (Full)**: Advanced mathematical properties, monotonicity, boundary conditions
- **Level 3 (Release)**: Edge cases, numerical stability, extreme parameters
- **Level 4 (Paper)**: Research experiment validation, reproducibility

### Method Coverage
- **Basic**: Gaussian baseline, core allocation methods (decomposition, analytic, direct)
- **Full**: All allocation variants, other schemes (local, Poisson, shuffle), direction consistency
- **Release**: Combined methods, comprehensive coverage, type annotations
- **Paper**: Research experiments, paper-specific scenarios

### Integration Testing
- **Basic**: Individual function testing, core parameter validation
- **Full**: Cross-method comparisons, type checking, mathematical properties
- **Release**: Complete workflow validation, comprehensive edge cases
- **Paper**: End-to-end experiment reproduction, research validation

## Current Test Status

### Test Suite Health
- **Total Tests**: 90+ across all levels
- **Basic Level**: 27 tests - ✅ All passing
- **Full Level**: 37 tests - ✅ All passing  
- **Release Level**: 26 tests - ✅ All passing
- **Paper Level**: Variable - Excluded from standard test runs

### Recent Fixes
1. **Decomposition Bug Fixed**: NameError in `allocation_epsilon_decomposition_add` resolved
2. **File Renaming**: `test_release_03_complete_type_annotations.py` → `test_release_02_complete_type_annotations.py`
3. **Error Handling**: Verified no inappropriate error suppression in test suite

## Running Individual Test Files

```bash
# Run specific test files
pytest basic/test_basic_01_functionality.py -v
pytest full/test_full_04_type_annotations.py -v
pytest release/test_release_02_complete_type_annotations.py -v

# Run specific test methods
pytest basic/test_basic_01_functionality.py::TestGaussianMechanismBaseline::test_gaussian_epsilon_basic -v
pytest full/test_full_03_mathematical_properties.py::TestMathematicalConstraints -v
```

## Dependencies

Test dependencies are defined in project requirements:
- pytest >= 8.0.0
- numpy (for numerical testing)
- mypy (for type checking)
- Standard library modules

## Continuous Integration

For CI/CD pipelines, use:
```bash
# Fast feedback (every commit)
python run_tests.py basic

# Thorough testing (pull requests)  
python run_tests.py full

# Release validation (before release)
python run_tests.py release

# Complete validation (major releases)
python run_tests.py all
```

## Performance Notes

- Tests include runtime warnings for numerical overflow in RDP calculations
- All tests are designed to complete within reasonable time limits
- No inappropriate error suppression - all errors properly validate functionality 