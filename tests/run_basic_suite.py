#!/usr/bin/env python3
"""
Basic Test Suite Runner

Runs all tests in the basic suite and aggregates results into a single JSON file.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.test_utils import TestSuiteAggregator, run_test_file_and_get_results


def run_basic_suite(output_dir: str = "tests/test_results") -> str:
    """Run the basic test suite and return the path to the aggregated results"""
    
    suite_aggregator = TestSuiteAggregator("basic", output_dir)
    
    # Set environment variable to indicate suite run (so individual tests don't save JSON files)
    os.environ['PYTEST_SUITE_RUN'] = 'true'
    
    try:
        # Define test files in the basic suite
        basic_test_files = [
            "tests/basic/test_basic_01_functionality.py"
        ]
        
        print("Running Basic Test Suite...")
        print("=" * 50)
        
        for test_file in basic_test_files:
            print(f"\nRunning {test_file}...")
            test_file_path = project_root / test_file
            
            # Run the test file and get results
            results = run_test_file_and_get_results(str(test_file_path), output_dir)
            
            # Add to suite aggregator
            test_file_name = Path(test_file).stem
            suite_aggregator.add_test_file_results(test_file_name, results)
            
            # Print brief summary
            summary = results.get('summary', {})
            print(f"  Results: {summary.get('total_tests', 0)} tests "
                   f"({summary.get('passed', 0)} passed, {summary.get('failed', 0)} failed, "
                   f"{summary.get('errors', 0)} errors, {summary.get('skipped', 0)} skipped)")
        
        # Finalize and save suite results
        suite_results_path = suite_aggregator.finalize_and_save()
        
        print("\n" + "=" * 50)
        print("Basic Suite Summary:")
        print(suite_aggregator.get_summary_report())
        print(f"\nSuite results saved to: {suite_results_path}")
        
        return suite_results_path
        
    finally:
        # Clean up environment variable
        os.environ.pop('PYTEST_SUITE_RUN', None)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the basic test suite")
    parser.add_argument("--output-dir", default="tests/test_results", 
                       help="Output directory for test results")
    
    args = parser.parse_args()
    
    run_basic_suite(args.output_dir)
