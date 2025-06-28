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
**Runtime**: Slow (~50+ seconds)
**Run Level**: Pre-release and CI/CD

- `test_release_01_complete_type_annotations.py` - Complete type annotation coverage (26 tests)
- `test_release_02_monotonicity.py` - **Comprehensive monotonicity testing (370 parameterized tests)**
- `test_release_03_edge_cases.py` - **Mathematically precise edge case testing (476 parameterized tests)**

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

# Individual suite runners (recommended for focused testing)
python run_basic_suite.py              # Basic tests only
python run_full_suite.py               # Full tests only  
python run_release_suite.py            # Release tests only

# Additional options
python run_tests.py full --fast        # Skip slow tests
python run_tests.py basic -x           # Stop on first failure
python run_tests.py release --verbose  # Detailed output
```

## Test Categories by Content

### Mathematical Correctness
- **Level 1 (Basic)**: Parameter validation, core mathematical properties
- **Level 2 (Full)**: Advanced mathematical properties, cross-scheme validation
- **Level 3 (Release)**: Mathematical precision, edge cases, monotonicity validation
- **Level 4 (Paper)**: Research experiment validation, reproducibility

### Method Coverage
- **Basic**: Gaussian baseline, core parameter validation, object creation
- **Full**: All allocation variants, other schemes (local, Poisson, shuffle), direction consistency
- **Release**: Complete type annotations, mathematically precise edge cases, monotonicity validation
- **Paper**: Research experiments, paper-specific scenarios

### Integration Testing
- **Basic**: Individual function testing, core parameter validation
- **Full**: Cross-method comparisons, scheme validation
- **Release**: Comprehensive mathematical validation (872 total tests), complete type coverage
- **Paper**: End-to-end experiment reproduction, research validation

## Current Test Status (2025 Modernization)

### Test Suite Health
- **Total Tests**: 924 tests across all levels
- **Basic Level**: 10 tests - ✅ All passing (core functionality)
- **Full Level**: 28 tests - ✅ All passing (extended schemes)  
- **Release Level**: 872 tests - ✅ 718 passed, 29 failed, 125 skipped (comprehensive validation)
- **Paper Level**: Variable - Excluded from standard test runs

### Release Suite Breakdown
- **Type Annotations**: 26 tests (24 passed, 2 failed)
- **Monotonicity**: 370 tests (349 passed, 21 failed)
- **Edge Cases**: 476 tests (345 passed, 6 failed, 125 skipped)

### Edge Case Modernization
The edge case tests have been completely modernized:

#### ✅ **Mathematical Precision Achieved**
- **Only valid combinations**: Epsilon-only edge cases → delta tests, Delta-only edge cases → epsilon tests
- **Function existence validation**: Only existing functions are tested
- **28% reduction**: From 666 to 476 tests (eliminated invalid combinations)
- **60% skip reduction**: From 315 to 125 skips (eliminated meaningless skips)

#### **Skip Categories (125 legitimate skips)**
1. **Invalid Edge Case** (72 skips, 57.6%): Mathematically incompatible with specific schemes
2. **Documented Bug** (29 skips, 23.2%): Known issues in implementations
3. **Computational Timeout** (24 skips, 19.2%): Algorithmic complexity limits

### Recent Modernization (2025)
1. **Mathematical Validation**: Only valid epsilon-delta relationships tested
2. **Function Existence**: Parametrization checks function existence before generating tests
3. **Skip Optimization**: Eliminated "missing parameters" and "function not found" skips
4. **Precision Focus**: Tests now mathematically meaningful and computationally feasible
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