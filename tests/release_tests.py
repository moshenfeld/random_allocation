#!/usr/bin/env python
"""
Release tests for the Random Allocation project.

This script runs all tests that should pass before a release, including:
1. Basic functionality tests
2. Data handler tests
"""

import os
import sys
import subprocess
import time

# Set the matplotlib backend to 'Agg' to prevent plots from being displayed
os.environ['MPLBACKEND'] = 'Agg'

def run_command(cmd: list, name: str) -> bool:
    """Run a command and return True if it succeeds."""
    print(f"\n{'-' * 80}")
    print(f"Running {name}...")
    print(f"{'-' * 80}")
    
    start_time = time.time()
    # Capture output to display it properly
    result = subprocess.run(cmd, text=True, capture_output=True, env=dict(os.environ, MPLBACKEND='Agg'))
    elapsed_time = time.time() - start_time
    
    # Always print the output so that test failures are visible
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    print(f"\n{'-' * 80}")
    if result.returncode == 0:
        print(f"✅ {name} passed in {elapsed_time:.2f} seconds")
        return True
    else:
        print(f"❌ {name} failed in {elapsed_time:.2f} seconds")
        return False

def main() -> None:
    """Run all release tests."""
    print("Running release tests for Random Allocation project")
    
    # Ensure proper Python path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.environ['PYTHONPATH'] = project_root
    
    # List of test commands to run
    tests = [
        {
            "name": "Basic Tests",
            "cmd": [sys.executable, os.path.join(project_root, "tests", "basic_tests.py")]
        },
        {
            "name": "Data Handler Tests",
            "cmd": [sys.executable, os.path.join(project_root, "tests", "data_handler_tests.py")]
        }
    ]
    
    # Run all tests and track results
    results = []
    for test in tests:
        success = run_command(test["cmd"], test["name"])
        results.append((test["name"], success))
    
    # Print summary
    print("\n" + "=" * 80)
    print("RELEASE TEST SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {name}")
        all_passed = all_passed and success
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main() 