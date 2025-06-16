# Test Structure and Organization

## Hierarchical Test Structure

The tests are organized in a three-tier hierarchy:

### 🔵 BASIC Tests (`basic/`)
**Purpose**: Core functionality validation
**Runtime**: Fast (~1-3 seconds)
**Run Level**: Always included

- `test_basic_01_functionality.py` - Parameter validation, basic operations
- `test_basic_02_core_allocation_methods.py` - Core allocation scheme testing

### 🟡 FULL Tests (`full/`)  
**Purpose**: Comprehensive feature testing
**Runtime**: Moderate (~5-15 seconds)
**Run Level**: Development and pre-release

- `test_full_01_additional_allocation_methods.py` - Extended allocation methods
- `test_full_02_other_schemes.py` - Non-allocation privacy schemes
- `test_full_03_mathematical_properties.py` - Mathematical correctness
- `test_full_04_type_annotations.py` - Type annotation compliance

### 🔴 RELEASE Tests (`release/`)
**Purpose**: Integration and end-to-end validation  
**Runtime**: Slow (~15+ seconds)
**Run Level**: Pre-release and CI/CD

- `test_release_01_comprehensive_coverage.py` - Complex scenarios and edge cases
- `test_release_02_paper_experiments.py` - Paper experiment integration tests

## Test Runner Usage

```bash
# Run only basic tests (fast feedback during development)
python run_tests.py basic

# Run basic + full tests (thorough development testing)
python run_tests.py full

# Run all tests (complete validation for release)
python run_tests.py release
# or
python run_tests.py all

# Additional options
python run_tests.py full --fast        # Skip slow tests
python run_tests.py basic -x           # Stop on first failure
```

## Test Categories by Content

### Mathematical Correctness
- **Level 1 (Basic)**: Parameter validation, basic mathematical properties
- **Level 2 (Full)**: Advanced mathematical properties, monotonicity
- **Level 3 (Release)**: Edge cases, numerical stability

### Method Coverage
- **Basic**: Gaussian baseline, core allocation decomposition
- **Full**: All allocation variants, other schemes (local, poisson, shuffle)
- **Release**: Combined methods, recursive calculations, extreme parameters

### Integration Testing
- **Basic**: Individual function testing
- **Full**: Cross-method comparisons, type checking
- **Release**: Paper experiments, complete workflow validation

## Bug Discovery Results

### Known Issues Identified
1. **Zero Epsilon Bug** (Critical): `allocation_epsilon_decomposition` returns ε = 0.0
2. **Infinite Epsilon Bug** (Critical): Conservative parameters still return ε = ∞
3. **Design Constraints**: 
   - Shuffle method: only supports `num_selected=1`
   - Lower bound method: only supports `num_selected=1`

### Test Coverage Summary
- **Total Tests**: 49+ across all levels
- **Basic Level**: 8 tests - ✅ All passing
- **Full Level**: 30+ tests - 🟡 Mostly passing with known issues
- **Release Level**: 10+ tests - 🔴 Some failures due to known bugs

## Running Individual Test Files

```bash
# Run specific test files
pytest basic/test_basic_01_functionality.py -v
pytest full/test_full_04_type_annotations.py -v
pytest release/test_release_02_paper_experiments.py -v

# Run specific test methods
pytest basic/test_basic_01_functionality.py::TestBasicFunctionality::test_privacy_params_validation -v
```

## Dependencies

See `test_requirements.txt` for testing dependencies:
- pytest >= 8.4.0
- numpy (for numerical testing)
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
``` 