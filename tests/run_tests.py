#!/usr/bin/env python3
"""
Test Runner for Random Allocation Project

Runs tests in hierarchical order: basic -> full -> release

Usage:
    python run_tests.py basic    # Run only basic tests
    python run_tests.py full     # Run basic + full tests  
    python run_tests.py release  # Run all tests (basic + full + release)
    python run_tests.py all      # Same as release
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_pytest(test_dirs, extra_args=None):
    """Run pytest on specified directories"""
    cmd = ["python", "-m", "pytest", "-v"]
    
    if extra_args:
        cmd.extend(extra_args)
    
    cmd.extend(test_dirs)
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run tests at different levels")
    parser.add_argument("level", choices=["basic", "full", "release", "all"], 
                       help="Test level to run")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    parser.add_argument("--fast", action="store_true",
                       help="Skip slow tests (exclude 'slow' marker)")
    parser.add_argument("--stopfirst", "-x", action="store_true",
                       help="Stop on first failure")
    
    args = parser.parse_args()
    
    # Build pytest arguments
    extra_args = []
    if args.fast:
        extra_args.extend(["-m", "not slow"])
    if args.stopfirst:
        extra_args.append("-x")
    
    # Determine which test directories to run
    test_dirs = []
    
    if args.level in ["basic", "full", "release", "all"]:
        test_dirs.append("basic/")
        print("🔵 Including BASIC tests")
    
    if args.level in ["full", "release", "all"]:
        test_dirs.append("full/")
        print("🟡 Including FULL tests")
    
    if args.level in ["release", "all"]:
        test_dirs.append("release/")
        print("🔴 Including RELEASE tests")
    
    print(f"\n{'='*60}")
    print(f"Running {args.level.upper()} test suite")
    print(f"Test directories: {test_dirs}")
    if extra_args:
        print(f"Extra args: {extra_args}")
    print(f"{'='*60}\n")
    
    # Run the tests
    exit_code = run_pytest(test_dirs, extra_args)
    
    # Report results
    if exit_code == 0:
        print(f"\n✅ {args.level.upper()} tests PASSED")
    else:
        print(f"\n❌ {args.level.upper()} tests FAILED (exit code: {exit_code})")
    
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 