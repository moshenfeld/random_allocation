# Random Allocation for Differential Privacy - Test Documentation

This document details the comprehensive four-tier test suite for the Random Allocation project.

## Four-Tier Testing System

The project uses a four-tier testing approach for different development and deployment scenarios:

### **BASIC Tests** (< 5 seconds)
**Purpose**: Essential functionality checks to catch obvious mistakes  
**When to use**: Quick development feedback, pre-commit checks  
**Command**: `python tests/run_tests.py basic` or `python tests/run_basic_suite.py`

- **Parameter Validation**: Core PrivacyParams class functionality
- **Core Functionality**: Essential privacy algorithm verification
- **Basic Mathematical Properties**: Fundamental correctness checks

### **FULL Tests** (< 30 seconds)  
**Purpose**: Comprehensive validation including mathematical correctness and type checking  
**When to use**: Before merging, development milestones  
**Command**: `python tests/run_tests.py full` or `python tests/run_full_suite.py`

- All BASIC tests plus:
- **Extended Methods**: Additional allocation methods and direction consistency
- **Other Schemes**: Cross-method validation with local, Poisson, and shuffle schemes
- **Mathematical Properties**: Advanced correctness and boundary condition testing
- **Type Annotations**: Static type checking and compliance validation

### **RELEASE Tests** (< 60 seconds)
**Purpose**: Complete validation for production readiness  
**When to use**: Before releases, major deployments  
**Command**: `python tests/run_tests.py release` or `python tests/run_release_suite.py`

- All FULL tests plus:
- **Comprehensive Coverage**: Mathematically precise edge cases and monotonicity validation
- **Complete Type Validation**: Exhaustive type annotation coverage (26 tests)
- **Edge Case Testing**: 476 mathematically valid boundary condition tests
- **Monotonicity Validation**: 370 mathematical property tests

### **PAPER Tests** (Variable timing)
**Purpose**: Research reproducibility and paper experiment validation  
**When to use**: Research validation, paper submission, reproducibility checks  
**Command**: `python tests/run_tests.py paper`

- All RELEASE tests plus:
- **Research Experiments**: Bit-exact reproducibility of paper results
- **Experiment Validation**: Complete paper experiment coverage

## Test Directory Structure

```
tests/
├── run_tests.py                    # Four-tier hierarchical test runner
├── run_basic_suite.py             # Basic suite runner
├── run_full_suite.py              # Full suite runner  
├── run_release_suite.py           # Release suite runner
├── basic/                          # Basic functionality tests (10 tests)
│   └── test_basic_01_functionality.py        # Parameter validation, Gaussian mechanism
├── full/                           # Comprehensive tests (28 tests)
│   ├── test_full_01_additional_allocation_methods.py  # Extended methods
│   ├── test_full_02_other_schemes.py                  # Other schemes  
│   └── test_full_03_utility_functions.py             # Core utility functions
├── release/                        # Release validation tests (872 tests)
│   ├── test_release_01_complete_type_annotations.py  # Complete type validation (26 tests)
│   ├── test_release_02_monotonicity.py               # Monotonicity validation (370 tests)
│   └── test_release_03_edge_cases.py                 # Mathematically precise edge cases (476 tests)
├── paper/                          # Research tests (Variable)
│   └── test_paper_01_experiments.py                  # Paper experiments
├── README.md                       # Test suite overview
└── TEST_STRUCTURE.md              # Detailed test organization
```

## Test File Details

### `basic/test_basic_01_functionality.py` (10 tests)
**Purpose**: Validate core functionality and parameter handling  
**Key Tests**:
- Gaussian mechanism baseline verification
- PrivacyParams class creation and validation
- Direction enum functionality
- Basic parameter boundary checks

### `full/test_full_01_additional_allocation_methods.py`
**Purpose**: Extended allocation algorithm functionality verification  
**Key Tests**:
- Additional allocation methods beyond core functionality
- Direction consistency across methods
- Extended scheme validation
- Cross-method compatibility

### `full/test_full_02_other_schemes.py`
**Purpose**: Non-allocation privacy scheme validation
**Key Tests**:
- Local differential privacy schemes
- Poisson mechanism testing
- Shuffle privacy validation
- Cross-scheme comparison

### `full/test_full_03_utility_functions.py`
**Purpose**: Core utility function validation
**Key Tests**:
- Search functions and parameter bounds
- Convergence validation utilities
- Mathematical helper functions
- Utility function integration

### `release/test_release_01_complete_type_annotations.py` (26 tests)
**Purpose**: Complete type annotation coverage validation  
**Key Tests**:
- Function return type validation across all modules
- Type conversion testing and compliance
- Callable type annotations verification
- Optional and union type handling
- Complete module coverage validation

### `release/test_release_02_monotonicity.py` (370 tests)
**Purpose**: Comprehensive mathematical monotonicity validation  
**Key Tests**:
- Parameter monotonicity across all allocation schemes
- Direction consistency validation
- Mathematical property preservation
- Cross-scheme monotonicity comparison
- Boundary condition monotonicity

### `release/test_release_03_edge_cases.py` (476 tests)
**Purpose**: Mathematically precise edge case validation  
**Key Tests**:
- **Epsilon Edge Cases**: Valid epsilon-only scenarios → delta tests (217 tests)
- **Delta Edge Cases**: Valid delta-only scenarios → epsilon tests (259 tests)
- **Mathematical Precision**: Only valid epsilon-delta relationships tested
- **Function Existence**: Only existing functions tested (no missing function skips)
- **Boundary Conditions**: Extreme parameter validation with computational limits

#### Edge Case Categories:
- **Invalid Edge Case** (72 skips): Mathematically incompatible parameter combinations
- **Documented Bug** (29 skips): Known implementation issues  
- **Computational Timeout** (24 skips): Algorithmic complexity limits

### `paper/test_paper_01_experiments.py`
**Purpose**: Research reproducibility and paper experiment validation
- Error condition handling

### `release/test_release_02_complete_type_annotations.py` (15 tests)
**Purpose**: Complete type annotation coverage and validation  
**Key Tests**:
- Type alias compliance
- Constant type annotations
- Function signature completeness
- Runtime type validation
- MyPy integration testing

### `paper/test_paper_01_experiments.py` (Variable tests)
**Purpose**: Research reproducibility and paper experiment validation  
**Key Tests**:
- **Research Experiment Reproduction**: Validates exact reproduction of paper results
- **Data Integrity**: Ensures experimental data consistency
- **Reproducibility Verification**: Tests deterministic result generation

## Usage Instructions

### Hierarchical Testing
```bash
# Fast development feedback (< 5 seconds)
python tests/run_tests.py basic

# Comprehensive development testing (< 30 seconds)
python tests/run_tests.py full

# Complete release validation (< 60 seconds)
python tests/run_tests.py release

# Full validation including research experiments
python tests/run_tests.py paper

# Individual suite runners (recommended for focused testing)
python tests/run_basic_suite.py     # Basic tests only
python tests/run_full_suite.py      # Full tests only
python tests/run_release_suite.py   # Release tests only
```

### Advanced Options
```bash
# Skip slow tests
python tests/run_tests.py full --fast

# Stop on first failure
python tests/run_tests.py basic -x

# Verbose output with details
python tests/run_tests.py release --verbose
```

### Individual Test Execution
```bash
# Run specific test directories
pytest tests/basic/ -v
pytest tests/full/test_full_03_utility_functions.py -v

# Run specific test classes or methods
pytest tests/basic/test_basic_01_functionality.py::TestGaussianMechanismBaseline -v
pytest -k "epsilon" -v  # Run tests matching pattern

# Run release tests individually
pytest tests/release/test_release_03_edge_cases.py -v  # Edge cases only
pytest tests/release/test_release_02_monotonicity.py -v  # Monotonicity only
```

## Test Quality Metrics (2025 Status)

- **Total Tests**: 924 tests across all tiers
- **Mathematical Precision**: Edge cases use only valid epsilon-delta relationships
- **Function Validation**: Only existing functions tested (no missing function skips)
- **Performance**: All tests complete within designated time limits
- **Reliability**: Deterministic results with appropriate numerical tolerance
- **Type Safety**: Complete type annotation coverage and validation

### Release Suite Breakdown (872 tests)
- **Type Annotations**: 26 tests (24 passed, 2 failed) - 92.3% success
- **Monotonicity**: 370 tests (349 passed, 21 failed) - 94.3% success  
- **Edge Cases**: 476 tests (345 passed, 6 failed, 125 skipped) - 72.5% success

### Edge Case Modernization
- **28% test reduction**: From 666 to 476 tests (eliminated invalid combinations)
- **60% skip reduction**: From 315 to 125 skips (eliminated meaningless skips)
- **Mathematical precision**: Only valid mathematical relationships tested
- **Function existence**: Parametrization validates function existence

## Current Test Status

✅ **Modernized test suite** - The complete test suite validates:
- Mathematical correctness of all privacy allocation methods with precision
- Comprehensive edge case validation with legitimate boundary conditions
- Complete type annotation coverage and compliance
- Research experiment reproducibility

### Recent Modernization (2025)
- **Mathematical Validation**: Edge cases now use correct epsilon-delta relationships
- **Function Existence Checks**: Parametrization validates function existence before testing
- **Skip Optimization**: Eliminated Category 1 ("missing parameters") skips
- **Precision Focus**: Tests are mathematically meaningful and computationally feasible

## Maintenance Guidelines

### Adding New Tests
1. **Categorize**: Determine appropriate tier (BASIC/FULL/RELEASE/PAPER)
2. **File Placement**: Add to existing files or create focused new files
3. **Mathematical Validity**: Ensure edge cases use valid parameter relationships
4. **Function Existence**: Verify functions exist before parametrization
5. **Documentation**: Update this document with new test descriptions

### Modifying Existing Tests
- **Preserve Mathematical Intent**: Maintain mathematical correctness and relationships
- **Function Validation**: Check function existence in parametrization
- **Update Documentation**: Reflect changes in test descriptions
- **Verify Performance**: Ensure modifications don't exceed time limits
- **Test Isolation**: Maintain independence between test cases