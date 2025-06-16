# Random Allocation for Differential Privacy - Test Documentation

This document details the comprehensive three-tier test suite for the Random Allocation project.

## Three-Tier Testing System

The project uses a three-tier testing approach for different development and deployment scenarios:

### **BASIC Tests** (< 1 minute)
**Purpose**: Essential functionality checks to catch obvious mistakes  
**When to use**: Quick development feedback, pre-commit checks  
**Command**: `python tests/run_test_suite.py basic`

- **Parameter Validation**: Core PrivacyParams class functionality
- **Core Functionality**: Essential privacy algorithm verification

### **FULL Tests** (< 5 minutes)  
**Purpose**: Comprehensive validation including mathematical correctness and type checking  
**When to use**: Before merging, development milestones  
**Command**: `python tests/run_test_suite.py full`

- All BASIC tests plus:
- **Mathematical Properties**: Privacy guarantees and numerical correctness
- **Edge Cases**: Robustness testing with extreme parameters
- **Scheme Comparison**: Cross-method validation
- **Integration Tests**: End-to-end workflow validation (including data handling)
- **Type Annotations**: Static type checking with mypy

### **RELEASE Tests** (< 2 hours)
**Purpose**: Complete validation for production readiness  
**When to use**: Before releases, major deployments  
**Command**: `python tests/run_test_suite.py release`

- All FULL tests plus:
- **Performance Tests**: Benchmarking and regression detection
- **Paper Experiments**: Bit-exact reproducibility of research results

## Test Directory Structure

```
tests/
├── conftest.py                     # Pytest configuration and shared fixtures
├── run_test_suite.py               # Three-tier test runner
├── test_privacy_params.py          # Parameter validation (BASIC)
├── test_basic_functionality.py     # Core algorithm tests (BASIC)
├── test_mathematical_properties.py # Mathematical correctness (FULL)
├── test_edge_cases.py              # Robustness testing (FULL)
├── test_scheme_comparison.py       # Cross-method validation (FULL)
├── test_integration.py             # End-to-end + data handling (FULL)
├── test_type_checking.py           # Static type validation (FULL)
├── test_performance.py             # Benchmarking (RELEASE)
├── test_paper_experiments.py       # Research reproducibility (RELEASE)
├── test_requirements.txt           # Test-specific dependencies
├── cleanup.sh                      # Cleanup script for test artifacts
├── README.md                       # Test directory documentation
└── __init__.py                     # Python package marker
```

## Test File Details

### `test_privacy_params.py` (BASIC - 22 tests)
**Purpose**: Validate PrivacyParams class and parameter handling  
**Key Tests**:
- Parameter creation and validation
- Invalid parameter rejection
- Boundary value handling
- Type consistency checks

### `test_basic_functionality.py` (BASIC - ~20 tests)
**Purpose**: Essential algorithm functionality verification  
**Key Tests**:
- Core privacy algorithms (local, Poisson PLD, allocation methods)
- Direction handling (ADD/REMOVE/BOTH)
- Basic parameter validation
- Numerical consistency checks
- Algorithm output validity

### `test_mathematical_properties.py` (FULL - 27 tests)  
**Purpose**: Validate mathematical properties and correctness  
**Key Tests**:
- Privacy parameter bounds (δ ∈ [0,1], ε ≥ 0)
- Monotonicity relationships (sigma, num_steps, num_selected, num_epochs)
- Lower bound validations
- Cross-method consistency
- Numerical precision handling

### `test_edge_cases.py` (FULL - ~15 tests)
**Purpose**: Robustness testing with extreme parameters  
**Key Tests**:
- Extreme parameter values
- Boundary conditions
- Performance under stress
- Memory usage validation
- Timeout handling

### `test_scheme_comparison.py` (FULL - ~12 tests)
**Purpose**: Cross-method validation and comparison  
**Key Tests**:
- Method agreement within tolerance
- Relative performance characteristics  
- Parameter compatibility across methods
- Ordering relationships between schemes

### `test_integration.py` (FULL - ~18 tests)
**Purpose**: End-to-end workflow validation  
**Key Tests**:
- README example reproduction
- Complete algorithm workflows
- Data handling (save/load functionality)
- Parameter combination validation  
- Error handling and recovery
- Configuration testing

### `test_type_checking.py` (FULL - 4 tests)
**Purpose**: Static type checking and annotation validation  
**Key Tests**:
- MyPy type error detection
- Type annotation coverage measurement
- Type consistency validation
- Enum type verification

### `test_performance.py` (RELEASE - ~15 tests)
**Purpose**: Performance benchmarking and regression detection  
**Key Tests**:
- Method timing benchmarks
- Memory usage monitoring
- Performance regression detection
- Scalability testing
- Resource utilization validation

### `test_paper_experiments.py` (RELEASE - 7 tests)
**Purpose**: Bit-exact reproducibility of research experiments  
**Key Tests**:
- **Bit-Exact Validation**: Ensures research experiments produce identical results across runs
- **File Integrity**: Validates data file consistency
- **Experiment Coverage**: Tests all 7 paper experiments
- **Automated Cleanup**: Manages temporary files and backups

**Bit-Exact Testing Process**:
1. **Backup & Move**: Creates backups of existing data files and moves originals to force recomputation
2. **Recompute**: Runs experiments without cached data to generate fresh results
3. **Compare**: Performs binary file comparison for exact matching
4. **Cleanup/Restore**: Removes backups if identical, restores originals if different
5. **Investigation**: Provides detailed analysis of any differences found

## Usage Instructions

### Quick Testing
```bash
# Fast development feedback
python tests/run_test_suite.py basic

# Before committing/merging  
python tests/run_test_suite.py full --verbose

# Before releases
python tests/run_test_suite.py release --coverage
```

### Advanced Options
```bash
# Continue on failure
python tests/run_test_suite.py full --continue-on-failure

# Generate coverage report
python tests/run_test_suite.py full --coverage

# Verbose output with details
python tests/run_test_suite.py basic --verbose

# Custom report file
python tests/run_test_suite.py full --report custom_report.txt
```

### Individual Test Files
```bash
# Run specific test file
pytest tests/test_privacy_params.py -v

# Run with timeout and coverage
pytest tests/test_mathematical_properties.py --timeout=300 --cov=random_allocation
```

## Test Quality Metrics

- **Total Tests**: 110+ tests across all categories
- **Code Coverage**: Aims for >90% line coverage on core modules
- **Type Coverage**: Requires >70% type annotation coverage
- **Performance**: All tests complete within designated time limits
- **Reliability**: Deterministic results with <1e-10 numerical tolerance

## Maintenance Guidelines

### Adding New Tests
1. **Categorize**: Determine appropriate tier (BASIC/FULL/RELEASE)
2. **File Placement**: Add to existing files or create focused new files
3. **Timing**: Ensure tests complete within tier time limits
4. **Documentation**: Update this document with new test descriptions

### Modifying Existing Tests
1. **Backward Compatibility**: Maintain existing test interfaces
2. **Performance**: Monitor and optimize slow tests
3. **Reliability**: Ensure deterministic, non-flaky behavior
4. **Documentation**: Update documentation for significant changes

### Performance Optimization
- Use fixtures for expensive setup/teardown
- Parameterize similar tests to reduce duplication
- Mock external dependencies when appropriate
- Profile slow tests and optimize bottlenecks

### Debugging Failed Tests
1. **Isolation**: Run individual failing tests with `-v` flag
2. **Timing**: Check if failures are timeout-related
3. **Environment**: Verify test environment and dependencies
4. **Logs**: Examine detailed output for error patterns
5. **Bisection**: Use git bisect for regression identification

The test suite provides comprehensive validation while maintaining fast feedback loops for different development scenarios.

## Type Checking System

The project implements a sophisticated type checking system that follows the `type_annotations_guide` standards:

### Type Analysis Features

- **Module Discovery**: Automatically finds all Python modules in the project
- **Exclusion Handling**: Properly excludes external dependencies and legacy code
- **Coverage Calculation**: Provides detailed per-module type coverage statistics
- **Error Categorization**: Groups errors by type (var-annotated, no-redef, etc.)
- **Compliance Reporting**: Validates adherence to type annotation standards

### Release Standards

- **Minimum Coverage**: 80% of checkable modules must be fully typed
- **Error Tolerance**: ≤10 minor type errors allowed for release
- **Core Requirements**: All core data structures and public APIs must be typed
- **External Exclusions**: Dependencies and `*_external` modules are excluded

### Comprehensive Analysis Report

The release type checking provides detailed analysis including:

- Total modules discovered vs. checkable vs. fully typed
- Compliance with type_annotations_guide standards
- List of fully typed modules (success stories)
- Modules requiring attention with severity levels
- Detailed error type breakdown
- Specific recommendations for improvement

### Usage Examples

```bash
# Development type checking (permissive)
python tests/run_test_suite.py full

# Release type checking (strict)
python tests/run_test_suite.py release

# Direct mypy analysis
mypy random_allocation --show-error-codes
```