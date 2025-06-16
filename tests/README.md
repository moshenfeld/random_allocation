# Random Allocation Test Suite

This directory contains comprehensive tests for the random_allocation package.

## Quick Start

```bash
# Run all tests
python run_tests.py release

# Run basic tests only (fast)
python run_tests.py basic

# Run with pytest directly
pytest -v
```

## Documentation

- **📊 Latest Results**: [`COMPREHENSIVE_TEST_REPORT.md`](COMPREHENSIVE_TEST_REPORT.md) - Detailed test execution results
- **🏗️ Test Organization**: [`TEST_STRUCTURE.md`](TEST_STRUCTURE.md) - Current hierarchy and structure  
- **📈 Updated Report**: [`../UPDATED_TEST_REPORT.md`](../UPDATED_TEST_REPORT.md) - Most recent comprehensive analysis
- **📋 Summary**: [`../WORKING_VS_FAILING_SUMMARY.md`](../WORKING_VS_FAILING_SUMMARY.md) - Quick status overview

## Test Levels

- **Basic** (`basic/`): Core functionality, fast execution (~1-3s)
- **Full** (`full/`): Comprehensive testing (~5-15s) 
- **Release** (`release/`): Integration and end-to-end tests (~15+s)

## Current Status

The test suite exposes real mathematical bugs in the allocation algorithms. See the comprehensive report for detailed analysis of critical issues that need attention. 