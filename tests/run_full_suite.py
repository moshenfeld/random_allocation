#!/usr/bin/env python3
"""
Full Test Suite Runner

Runs all tests in the full test suite and aggregates results into a single JSON file.
"""

import os
import sys
from pathlib import Path
from typing import List

# Add the project root to sys.path to enable imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.test_utils import TestSuiteAggregator, run_test_file_and_get_results


def run_full_suite() -> None:
    """Run all tests in the full suite and save aggregated results."""
    # Set environment variable to indicate suite run (so individual tests don't save JSON files)
    os.environ['PYTEST_SUITE_RUN'] = 'true'
    
    try:
        # Define the test files in the full suite
        test_files = [
            "tests/full/test_full_01_additional_allocation_methods.py",
            "tests/full/test_full_02_other_schemes.py", 
            "tests/full/test_full_03_utility_functions.py"
        ]
        
        # Initialize the suite aggregator
        aggregator = TestSuiteAggregator("full_suite")
        
        # Run each test file and collect results
        for test_file in test_files:
            test_path = project_root / test_file
            if test_path.exists():
                print(f"Running {test_file}...")
                try:
                    results = run_test_file_and_get_results(str(test_path))
                    if results:
                        aggregator.add_test_file_results(test_file, results)
                        print(f"  ✓ {test_file} completed")
                    else:
                        print(f"  ⚠ {test_file} returned no results")
                except Exception as e:
                    print(f"  ✗ {test_file} failed: {e}")
                    # Add error result for failed test file
                    aggregator.add_test_file_results(test_file, {
                        'test_info': {'test_name': test_file, 'status': 'error'},
                        'summary': {'total_tests': 0, 'passed': 0, 'failed': 0, 'errors': 1, 'skipped': 0},
                        'categories': {},
                        'detailed_results': [{'test_id': f'{test_file}_error', 'status': 'error', 'error_message': str(e)}]
                    })
            else:
                print(f"  ⚠ {test_file} not found")
        
        # Save aggregated results
        output_path = aggregator.finalize_and_save()
        print(f"\nFull suite results saved to: {output_path}")
        
        # Print summary
        print("\nFull Suite Summary:")
        print(aggregator.get_summary_report())
        
    finally:
        # Clean up environment variable
        os.environ.pop('PYTEST_SUITE_RUN', None)


if __name__ == "__main__":
    run_full_suite()
