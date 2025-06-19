# Random Allocation for Differential Privacy - Test Documentation

This document details the comprehensive four-tier test suite for the Random Allocation project.

## Four-Tier Testing System

The project uses a four-tier testing approach for different development and deployment scenarios:

### **BASIC Tests** (< 5 seconds)
**Purpose**: Essential functionality checks to catch obvious mistakes  
**When to use**: Quick development feedback, pre-commit checks  
**Command**: `python tests/run_tests.py basic`

- **Parameter Validation**: Core PrivacyParams class functionality
- **Core Functionality**: Essential privacy algorithm verification
- **Basic Mathematical Properties**: Fundamental correctness checks

### **FULL Tests** (< 30 seconds)  
**Purpose**: Comprehensive validation including mathematical correctness and type checking  
**When to use**: Before merging, development milestones  
**Command**: `python tests/run_tests.py full`

- All BASIC tests plus:
- **Extended Methods**: Additional allocation methods and direction consistency
- **Other Schemes**: Cross-method validation with local, Poisson, and shuffle schemes
- **Mathematical Properties**: Advanced correctness and boundary condition testing
- **Type Annotations**: Static type checking and compliance validation

### **RELEASE Tests** (< 60 seconds)
**Purpose**: Complete validation for production readiness  
**When to use**: Before releases, major deployments  
**Command**: `python tests/run_tests.py release`

- All FULL tests plus:
- **Comprehensive Coverage**: Complex scenarios and edge cases
- **Complete Type Validation**: Exhaustive type annotation coverage
- **Integration Testing**: End-to-end workflow validation

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
├── basic/                          # Basic functionality tests (27 tests)
│   ├── test_basic_01_functionality.py        # Parameter validation, Gaussian mechanism (8 tests)
│   ├── test_basic_02_core_allocation_methods.py   # Core allocation methods (5 tests)
│   ├── test_basic_03_direct_allocation_add.py     # Direct allocation - add (3 tests)
│   ├── test_basic_03_direct_allocation_remove.py  # Direct allocation - remove (2 tests)
│   └── test_basic_04_direct_RDP_add.py            # RDP-based direct method (5 tests)
├── full/                           # Comprehensive tests (37 tests)
│   ├── test_full_01_additional_allocation_methods.py  # Extended methods (10 tests)
│   ├── test_full_02_other_schemes.py                  # Other schemes (12 tests)
│   ├── test_full_03_mathematical_properties.py       # Mathematical properties (8 tests)
│   └── test_full_04_type_annotations.py              # Type annotations (12 tests)
├── release/                        # Release validation tests (26 tests)
│   ├── test_release_01_comprehensive_coverage.py     # Edge cases (7 tests)
│   └── test_release_02_complete_type_annotations.py  # Complete type validation (15 tests)
├── paper/                          # Research tests (Variable)
│   └── test_paper_01_experiments.py                  # Paper experiments
├── README.md                       # Test suite overview
└── TEST_STRUCTURE.md              # Detailed test organization
```

## Test File Details

### `basic/test_basic_01_functionality.py` (8 tests)
**Purpose**: Validate core functionality and parameter handling  
**Key Tests**:
- Gaussian mechanism baseline verification
- PrivacyParams class creation and validation
- Direction enum functionality
- Basic parameter boundary checks

### `basic/test_basic_02_core_allocation_methods.py` (5 tests)  
**Purpose**: Essential allocation algorithm functionality verification  
**Key Tests**:
- Decomposition method testing
- Analytic method testing
- Method comparison and consistency
- Core mathematical property validation

### `basic/test_basic_03_direct_allocation_add.py` (3 tests)
**Purpose**: Direct allocation method validation (add direction)
**Key Tests**:
- Epsilon calculation with conservative parameters
- Delta calculation with conservative parameters
- Round-trip epsilon-delta consistency

### `basic/test_basic_03_direct_allocation_remove.py` (2 tests)
**Purpose**: Direct allocation method validation (remove direction)
**Key Tests**:
- Epsilon calculation for removal operations
- Delta calculation for removal operations

### `basic/test_basic_04_direct_RDP_add.py` (5 tests)
**Purpose**: RDP-based direct method validation
**Key Tests**:
- RDP epsilon calculation
- RDP delta calculation
- Round-trip validation
- Error handling for missing parameters

### `full/test_full_01_additional_allocation_methods.py` (10 tests)  
**Purpose**: Extended allocation methods and direction consistency  
**Key Tests**:
- Direct method with alpha orders
- RDP DCO method validation
- Direction consistency across methods
- Scheme configuration requirements

### `full/test_full_02_other_schemes.py` (12 tests)
**Purpose**: Non-allocation privacy schemes validation  
**Key Tests**:
- Local scheme testing (all directions)
- Poisson scheme PLD and RDP variants
- Shuffle scheme validation
- Inter-scheme comparison and ordering

### `full/test_full_03_mathematical_properties.py` (8 tests)
**Purpose**: Mathematical correctness and boundary conditions  
**Key Tests**:
- Parameter range validation
- Boundary condition testing
- Mathematical constraint verification
- Specific bug reproduction and fixes

### `full/test_full_04_type_annotations.py` (12 tests)
**Purpose**: Type annotation compliance and validation  
**Key Tests**:
- Function return type validation
- Type conversion testing
- Callable type annotations
- Optional and union type handling

### `release/test_release_01_comprehensive_coverage.py` (7 tests)
**Purpose**: Complex scenarios and edge case validation  
**Key Tests**:
- Remaining allocation methods
- Parameter boundary conditions
- Extreme precision requirements
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
pytest tests/full/test_full_03_mathematical_properties.py -v

# Run specific test classes or methods
pytest tests/basic/test_basic_01_functionality.py::TestGaussianMechanismBaseline -v
pytest -k "epsilon" -v  # Run tests matching pattern
```

## Test Quality Metrics

- **Total Tests**: 90+ tests across all tiers
- **Coverage**: Comprehensive validation of all privacy methods
- **Performance**: All tests complete within designated time limits
- **Reliability**: Deterministic results with appropriate numerical tolerance
- **Type Safety**: Extensive type annotation coverage and validation

## Current Test Status

✅ **All core tests passing** - The complete test suite validates:
- Mathematical correctness of all privacy allocation methods
- Comprehensive edge case and boundary condition handling  
- Complete type annotation coverage and compliance
- Research experiment reproducibility

### Recent Improvements
- **Fixed decomposition method bug**: Resolved NameError in epsilon calculation
- **Reorganized test structure**: Clear four-tier hierarchy with logical progression
- **Enhanced type validation**: Comprehensive type annotation testing
- **Improved documentation**: Clear test organization and purpose

## Maintenance Guidelines

### Adding New Tests
1. **Categorize**: Determine appropriate tier (BASIC/FULL/RELEASE/PAPER)
2. **File Placement**: Add to existing files or create focused new files
3. **Timing**: Ensure tests complete within tier time limits
4. **Documentation**: Update this document with new test descriptions

### Modifying Existing Tests
- **Preserve Intent**: Maintain original test purpose and validation goals
- **Update Documentation**: Reflect changes in test descriptions
- **Verify Performance**: Ensure modifications don't exceed time limits
- **Test Isolation**: Maintain independence between test cases