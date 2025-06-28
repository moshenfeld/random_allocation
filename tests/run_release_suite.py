#!/usr/bin/env python3
"""
Release Test Suite Runner

Runs all tests in the release test suite and aggregates results into a single JSON file.
"""

import os
import sys
from pathlib import Path
from typing import List

# Add the project root to sys.path to enable imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.test_utils import TestSuiteAggregator, run_test_file_and_get_results


def run_release_suite() -> None:
    """Run all tests in the release suite and save aggregated results."""
    # Set environment variable to indicate suite run (so individual tests don't save JSON files)
    os.environ['PYTEST_SUITE_RUN'] = 'true'
    
    try:
        # Define the test files in the release suite
        test_files = [
            "tests/release/test_release_01_complete_type_annotations.py",
            "tests/release/test_release_02_monotonicity.py",
            "tests/release/test_release_03_edge_cases.py"
        ]
        
        # Initialize the suite aggregator
        aggregator = TestSuiteAggregator("release_suite")
        
        # Run each test file and collect results
        total_files = len(test_files)
        for i, test_file in enumerate(test_files, 1):
            test_path = project_root / test_file
            if test_path.exists():
                print(f"[{i}/{total_files}] Running {test_file}...")
                print(f"  Progress: {i-1}/{total_files} files completed")
                try:
                    results = run_test_file_and_get_results(str(test_path))
                    if results:
                        aggregator.add_test_file_results(test_file, results)
                        summary = results.get('summary', {})
                        total = summary.get('total_tests', 0)
                        passed = summary.get('passed', 0)
                        failed = summary.get('failed', 0)
                        skipped = summary.get('skipped', 0)
                        print(f"  ✓ {test_file} completed: {total} tests ({passed} passed, {failed} failed, {skipped} skipped)")
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
        print(f"\nRelease suite results saved to: {output_path}")
        
        # Print summary
        print("\nRelease Suite Summary:")
        print(aggregator.get_summary_report())
        
    finally:
        # Clean up environment variable
        os.environ.pop('PYTEST_SUITE_RUN', None)


if __name__ == "__main__":
    run_release_suite()
