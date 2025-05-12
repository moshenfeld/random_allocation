#!/usr/bin/env python
"""
Type checking script for Random Allocation project.

This script runs mypy on the project with the appropriate configuration
and reports the status of type annotations in each module.
"""

import os
import sys
import subprocess
from typing import Dict, List, Tuple, Set, Optional
import argparse

def run_mypy(include_paths: Optional[List[str]] = None) -> Tuple[str, int]:
    """Run mypy on the project and return the output and exit code."""
    cmd = ["mypy"]
    
    if include_paths:
        cmd.extend(include_paths)
    else:
        cmd.append("random_allocation")
    
    # Add configurations
    cmd.extend([
        "--show-column-numbers",
        "--pretty",
        "--config-file=.mypy.ini"  # Explicitly use the mypy.ini config file
    ])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.returncode

def analyze_results(mypy_output: str) -> Dict[str, int]:
    """Analyze mypy output and calculate types per module."""
    modules: Dict[str, int] = {}
    
    # Modules to ignore based on .mypy.ini configuration
    ignored_modules = [
        "random_allocation.random_allocation_scheme.Monte_Carlo_external",
        "random_allocation.other_schemes.shuffle_external"
    ]
    
    for line in mypy_output.split('\n'):
        if ':' not in line:
            continue
        
        file_part = line.split(':')[0]
        if not file_part.endswith('.py'):
            continue
        
        # Extract module name
        module_path = file_part.replace('/', '.').replace('.py', '')
        if module_path.startswith('random_allocation.'):
            module_name = module_path
        else:
            continue
            
        # Skip modules that should be ignored
        if module_name in ignored_modules or any(module_name.endswith("_external") for module in ignored_modules if "_external" in module):
            continue
        
        # Count errors per module
        modules[module_name] = modules.get(module_name, 0) + 1
    
    return modules

def get_all_modules() -> Set[str]:
    """Find all Python modules in the project."""
    modules = set()
    
    # Modules to ignore based on .mypy.ini configuration
    ignored_modules = [
        "random_allocation.random_allocation_scheme.Monte_Carlo_external",
        "random_allocation.other_schemes.shuffle_external"
    ]
    
    for root, _, files in os.walk("random_allocation"):
        for file in files:
            if file.endswith(".py") and not file.startswith("__"):
                module_path = os.path.join(root, file)
                module_name = module_path.replace('/', '.').replace('.py', '')
                
                # Skip modules that should be ignored
                if module_name in ignored_modules or any(module_name.endswith("_external") for module in ignored_modules if "_external" in module):
                    continue
                    
                modules.add(module_name)
    
    return modules

def print_report(error_counts: Dict[str, int], all_modules: Set[str]) -> None:
    """Print a report of type annotation status."""
    print("\n=== Type Annotation Status Report ===\n")
    
    # Calculate fully typed modules
    fully_typed = all_modules - set(error_counts.keys())
    
    # Print fully typed modules
    print(f"Fully typed modules ({len(fully_typed)}/{len(all_modules)}):")
    for module in sorted(fully_typed):
        print(f"  ✓ {module}")
    
    # Print modules with type errors
    if error_counts:
        print("\nModules with type errors:")
        for module, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  ✗ {module}: {count} errors")
    
    # Print summary
    coverage = len(fully_typed) / len(all_modules) * 100 if all_modules else 0
    print(f"\nType annotation coverage: {coverage:.1f}%")
    total_errors = sum(error_counts.values())
    print(f"Total type errors: {total_errors}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Run type checking on the Random Allocation project")
    parser.add_argument("--path", "-p", nargs="+", help="Specific module paths to check")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print the full mypy output")
    args = parser.parse_args()
    
    print("Running mypy type checker...")
    mypy_output, exit_code = run_mypy(args.path)
    
    if args.verbose:
        print("\n=== mypy output ===\n")
        print(mypy_output)
    
    error_counts = analyze_results(mypy_output)
    all_modules = get_all_modules()
    
    print_report(error_counts, all_modules)
    
    # Exit with error code if type errors are found
    if error_counts:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main() 