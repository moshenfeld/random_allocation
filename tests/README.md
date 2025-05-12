# Random Allocation Test Suite

This directory contains tests for the random_allocation package. The tests are designed to ensure that refactoring doesn't break functionality.

## Running Tests

To run all tests:

```bash
# Run from the project root directory
python -m unittest discover tests
```

To run a specific test file:

```bash
python -m unittest tests.basic_tests
```

To run a specific test case:

```bash
python -m unittest tests.basic_tests.TestFunctionalityNotBroken
```

To run a specific test method:

```bash
python -m unittest tests.basic_tests.TestFunctionalityNotBroken.test_allocation_analytic_all_directions
```

## Test Design

The tests in `basic_tests.py` make basic calls to all functions in the codebase, checking that they run without errors. These are simple "smoke tests" that don't validate the correctness of results but ensure the functions can be called with reasonable parameters.

### Comprehensive Testing

The test suite tests all main functions with all supported direction options:

- `add`: Tests functions when adding elements to the dataset
- `remove`: Tests functions when removing elements from the dataset
- `both`: Tests functions with the combined add/remove scenario

Each test method uses `subTest` to clearly show which direction is being tested, making it easier to identify which specific configuration is failing if an issue arises.

### Parameters Tested

The test suite tests both epsilon and delta calculations for each algorithm:

- For epsilon calculation: Tests with a provided delta value
- For delta calculation: Tests with a provided epsilon value

Special parameters are used for algorithms with specific requirements (like Shuffle and Lower Bound schemes).

### Known Issues

The test suite includes handling for known issues in the algorithms. These are tracked and reported but don't cause test failures:

#### Infinity Values
Some functions return infinity for certain directions:
- `allocation_epsilon_analytic`: Returns infinity for all directions (add, remove, both)
- `allocation_epsilon_decomposition`: Returns infinity for 'add' and 'both' directions
- `allocation_epsilon_recursive`: Returns infinity for all directions (add, remove, both)

#### NaN Values
Some functions return NaN (Not a Number) values for certain directions:
- `allocation_epsilon_direct`: Returns NaN for 'remove' and 'both' directions

#### Warnings
Several functions generate runtime warnings:
- Overflow encountered in exp (RDP_DCO.py)
- Invalid value encountered in scalar divide (direct.py)
- Invalid value encountered in sqrt (local.py)
- "Potential alpha underflow" messages (multiple functions)

These warnings may indicate numerical stability issues in the algorithms.

### Warning Tracking

The test suite includes a `WarningCatcher` class that captures and summarizes warnings generated during tests. These warnings are reported at the end of each test to make them more visible.

## When Refactoring

When refactoring the code, run these tests to verify that you haven't broken basic functionality. When fixing the code, consider addressing:

1. The numerical stability issues that cause warnings
2. Cases where functions return infinity or NaN values
3. Consistent behavior across all direction options (add, remove, both)

If you fix any of the known issues, update the `KNOWN_ISSUES` dictionary in `basic_tests.py` to reflect the improvements. 