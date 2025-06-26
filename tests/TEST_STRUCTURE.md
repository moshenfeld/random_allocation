# Test Structure and Organization

## Hierarchical Test Structure

The tests are organized in a four-tier hierarchy:

### 🔵 BASIC Tests (`basic/`)
**Purpose**: Core functionality validation
**Runtime**: Fast (~1-5 seconds)
**Run Level**: Always included

- `test_basic_01_functionality.py` - Parameter validation, Gaussian mechanism, core objects

### 🟡 FULL Tests (`full/`)  
**Purpose**: Extended schemes and cross-method validation
**Runtime**: Moderate (~5-30 seconds)
**Run Level**: Development and pre-release

- `test_full_01_additional_allocation_methods.py` - Extended allocation methods, direction consistency
- `test_full_02_other_schemes.py` - Non-allocation privacy schemes (local, Poisson, shuffle)
- `test_full_03_utility_functions.py` - Core utility functions (search, bounds, convergence)

### 🔴 RELEASE Tests (`release/`)
**Purpose**: Comprehensive validation and integration testing
**Runtime**: Slow (~30+ seconds)
**Run Level**: Pre-release and CI/CD

- `test_release_01_complete_type_annotations.py` - Complete type annotation coverage and validation
- `test_release_02_monotonicity.py` - **Comprehensive monotonicity testing (370+ parameterized tests)**
- `test_release_03_edge_cases.py` - **Comprehensive edge case testing (666 parameterized tests)**

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
- **Basic**: Gaussian baseline, core parameter validation, object creation
- **Full**: All allocation variants, other schemes (local, Poisson, shuffle), direction consistency
- **Release**: Complete type annotations, comprehensive edge cases, monotonicity validation
- **Paper**: Research experiments, paper-specific scenarios

### Integration Testing
- **Basic**: Individual function testing, core parameter validation
- **Full**: Cross-method comparisons, scheme validation
- **Release**: Comprehensive mathematical validation (1000+ parameterized tests), complete type coverage
- **Paper**: End-to-end experiment reproduction, research validation

## Current Test Status

### Test Suite Health
- **Total Tests**: 58+ core tests plus 1000+ comprehensive parameterized tests
- **Basic Level**: 8 tests - ✅ All passing (core functionality)
- **Full Level**: 22 tests - ✅ All passing (extended schemes)
- **Release Level**: 28 tests - ✅ All passing (comprehensive validation)
- **Paper Level**: Variable - Excluded from standard test runs

### Comprehensive Test Coverage
- **Edge Cases**: 666 parameterized tests covering all allocation methods with extreme parameters
- **Monotonicity**: 370+ parameterized tests validating mathematical properties
- **Type Annotations**: Complete coverage across all modules
- **Cross-Scheme Validation**: All privacy schemes tested and compared

### Recent Cleanup (Current)
1. **Redundancy Removal**: Removed 6 redundant test files covered by comprehensive tests
2. **API Fixes**: Fixed validation method calls to match actual PrivacyParams API
3. **Consolidation**: Merged type annotation tests into single comprehensive suite
4. **Focus**: Streamlined to essential tests plus comprehensive edge/monotonicity coverage

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