#!/usr/bin/env python3
"""
Test Utilities - Shared functionality for test result reporting and JSON output
"""
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path


class ResultsReporter:
    """Utility class for collecting and reporting test results in JSON format"""
    
    def __init__(self, test_name: str, output_dir: str = "tests/test_results"):
        self.test_name = test_name
        self.output_dir = output_dir
        self.results = {
            'test_info': {
                'test_name': test_name,
                'timestamp': datetime.now().isoformat(),
                'start_time': time.time()
            },
            'summary': {
                'total_tests': 0,
                'passed': 0,
                'failed': 0,
                'errors': 0,
                'skipped': 0,
                'execution_time': 0.0
            },
            'categories': {},
            'detailed_results': []
        }
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_results_data(self) -> Dict[str, Any]:
        """Return the current results data structure"""
        return self.results.copy()
    
    def merge_results(self, other_results: Dict[str, Any]) -> None:
        """Merge results from another ResultsReporter or results dict"""
        
        # Update summary counts
        for key in ['total_tests', 'passed', 'failed', 'errors', 'skipped']:
            self.results['summary'][key] += other_results.get('summary', {}).get(key, 0)
        
        # Merge categories
        for category, counts in other_results.get('categories', {}).items():
            if category not in self.results['categories']:
                self.results['categories'][category] = {
                    'total': 0, 'passed': 0, 'failed': 0, 'errors': 0, 'skipped': 0
                }
            
            for status in ['total', 'passed', 'failed', 'errors', 'skipped']:
                self.results['categories'][category][status] += counts.get(status, 0)
        
        # Merge detailed results
        self.results['detailed_results'].extend(other_results.get('detailed_results', []))
    
    def add_test_result(self, 
                       test_id: str,
                       category: str,
                       status: str,  # 'passed', 'failed', 'error', 'skipped'
                       details: Dict[str, Any],
                       error_message: Optional[str] = None,
                       execution_time: Optional[float] = None):
        """Add a single test result"""
        
        # Update summary
        self.results['summary']['total_tests'] += 1
        self.results['summary'][status] += 1
        
        # Update category counts
        if category not in self.results['categories']:
            self.results['categories'][category] = {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'errors': 0,
                'skipped': 0
            }
        
        self.results['categories'][category]['total'] += 1
        self.results['categories'][category][status] += 1
        
        # Add detailed result
        result_entry = {
            'test_id': test_id,
            'category': category,
            'status': status,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        
        if error_message:
            result_entry['error_message'] = error_message
        
        if execution_time is not None:
            result_entry['execution_time'] = execution_time
            
        self.results['detailed_results'].append(result_entry)
    
    def finalize_and_save(self, save_to_file: bool = True) -> str:
        """Finalize results and optionally save to JSON file"""
        
        # Calculate total execution time
        self.results['test_info']['end_time'] = time.time()
        self.results['summary']['execution_time'] = (
            self.results['test_info']['end_time'] - self.results['test_info']['start_time']
        )
        
        if not save_to_file:
            return ""
        
        # Generate filename with timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.test_name}_results_{timestamp_str}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        return filepath
    
    def get_results(self) -> Dict[str, Any]:
        """Get the current results without saving to file"""
        # Finalize timing but don't save
        self.results['test_info']['end_time'] = time.time()
        self.results['summary']['execution_time'] = (
            self.results['test_info']['end_time'] - self.results['test_info']['start_time']
        )
        return self.results.copy()
    
    def get_summary_report(self) -> str:
        """Generate a human-readable summary report"""
        
        summary = self.results['summary']
        report_lines = [
            f"Test Results Summary for: {self.results['test_info']['test_name']}",
            f"Timestamp: {self.results['test_info']['timestamp']}",
            f"Execution Time: {summary['execution_time']:.2f} seconds",
            "",
            f"Total Tests: {summary['total_tests']}",
            f"Passed: {summary['passed']}",
            f"Failed: {summary['failed']}",
            f"Errors: {summary['errors']}",
            f"Skipped: {summary['skipped']}",
            "",
            "Results by Category:"
        ]
        
        for category, counts in self.results['categories'].items():
            report_lines.append(
                f"  {category}: {counts['total']} total "
                f"({counts['passed']} passed, {counts['failed']} failed, "
                f"{counts['errors']} errors, {counts['skipped']} skipped)"
            )
        
        return "\n".join(report_lines)


def load_and_report_results(json_filepath: str) -> str:
    """Load test results from JSON file and generate report"""
    
    with open(json_filepath, 'r') as f:
        results = json.load(f)
    
    # Create a temporary reporter to use the report generation method
    reporter = ResultsReporter("loaded_results")
    reporter.results = results
    
    return reporter.get_summary_report()


def compare_test_runs(json_filepath1: str, json_filepath2: str) -> str:
    """Compare two test runs and generate a comparison report"""
    
    with open(json_filepath1, 'r') as f:
        results1 = json.load(f)
    
    with open(json_filepath2, 'r') as f:
        results2 = json.load(f)
    
    report_lines = [
        f"Comparison between:",
        f"  Run 1: {results1['test_info']['test_name']} at {results1['test_info']['timestamp']}",
        f"  Run 2: {results2['test_info']['test_name']} at {results2['test_info']['timestamp']}",
        "",
        "Summary Changes:"
    ]
    
    summary1 = results1['summary']
    summary2 = results2['summary']
    
    for key in ['total_tests', 'passed', 'failed', 'errors', 'skipped']:
        change = summary2[key] - summary1[key]
        sign = '+' if change > 0 else ''
        report_lines.append(f"  {key}: {summary1[key]} → {summary2[key]} ({sign}{change})")
    
    exec_time_change = summary2['execution_time'] - summary1['execution_time']
    sign = '+' if exec_time_change > 0 else ''
    report_lines.append(f"  execution_time: {summary1['execution_time']:.2f}s → {summary2['execution_time']:.2f}s ({sign}{exec_time_change:.2f}s)")
    
    return "\n".join(report_lines)


class TestSuiteAggregator:
    """Aggregates results from multiple test files into a single suite result"""
    
    def __init__(self, suite_name: str, output_dir: str = "tests/test_results"):
        self.suite_name = suite_name
        self.output_dir = output_dir
        self.suite_reporter = ResultsReporter(suite_name, output_dir)
        self.test_files: List[Dict[str, Any]] = []
    
    def add_test_file_results(self, test_file_name: str, results_data: Dict[str, Any]) -> None:
        """Add results from a test file to the suite"""
        
        # Store reference to the test file results
        test_file_info = {
            'test_file': test_file_name,
            'results': results_data,
            'timestamp': results_data.get('test_info', {}).get('timestamp'),
            'execution_time': results_data.get('summary', {}).get('execution_time', 0.0),
            'summary': results_data.get('summary', {})
        }
        self.test_files.append(test_file_info)
        
        # Merge into suite aggregated results
        self.suite_reporter.merge_results(results_data)
        
        # Update suite metadata
        self.suite_reporter.results['test_info']['test_files'] = [
            {
                'name': tf['test_file'], 
                'timestamp': tf['timestamp'],
                'execution_time': tf['execution_time'],
                'summary': tf['summary']
            } 
            for tf in self.test_files
        ]
    
    def finalize_and_save(self) -> str:
        """Finalize the suite results and save to JSON"""
        
        # Update suite execution time as sum of all test files
        total_execution_time = sum(tf['execution_time'] for tf in self.test_files)
        self.suite_reporter.results['summary']['execution_time'] = total_execution_time
        
        # Add suite-specific metadata
        self.suite_reporter.results['test_info']['suite_name'] = self.suite_name
        self.suite_reporter.results['test_info']['num_test_files'] = len(self.test_files)
        
        return self.suite_reporter.finalize_and_save()
    
    def get_summary_report(self) -> str:
        """Generate a summary report for the entire suite"""
        
        lines = [
            f"Test Suite Results Summary: {self.suite_name}",
            f"Number of test files: {len(self.test_files)}",
            f"Total execution time: {sum(tf['execution_time'] for tf in self.test_files):.2f} seconds",
            "",
            "Test Files:"
        ]
        
        for tf in self.test_files:
            summary = tf['summary']
            lines.append(
                f"  {tf['test_file']}: {summary.get('total_tests', 0)} tests "
                f"({summary.get('passed', 0)} passed, {summary.get('failed', 0)} failed, "
                f"{summary.get('errors', 0)} errors, {summary.get('skipped', 0)} skipped) "
                f"[{tf['execution_time']:.2f}s]"
            )
        
        lines.extend(["", "Overall Suite Summary:"])
        lines.append(self.suite_reporter.get_summary_report())
        
        return "\n".join(lines)


def run_test_file_and_get_results(test_file_path: str, output_dir: str = "tests/test_results") -> Dict[str, Any]:
    """Run a single test file and return its results data structure"""
    
    import subprocess
    import sys
    import pytest
    from pathlib import Path
    
    # Extract test file name for reporting
    test_file_name = Path(test_file_path).stem
    
    # Custom pytest plugin to capture results from session
    class ResultsCapture:
        def __init__(self):
            self.results = None
        
        def pytest_sessionfinish(self, session, exitstatus):
            """Capture results from session after test completion"""
            if hasattr(session, '_capture_reporter'):
                self.results = session._capture_reporter.get_results()
    
    # Create results capture plugin
    results_capture = ResultsCapture()
    
    try:
        # Run the test file with pytest, using our custom plugin
        print(f"    Starting pytest execution for {test_file_name}...")
        
        # Set environment to indicate suite run (so no JSON files are created)
        env = os.environ.copy()
        env['PYTEST_SUITE_RUN'] = 'true'
        env['PYTEST_TEST_RESULTS_DIR'] = output_dir
        
        # Run pytest programmatically to capture session results
        exit_code = pytest.main([
            test_file_path,
            '-v',           # Verbose output to show individual test progress
            '--tb=line',    # Concise traceback format
            '--no-header',  # Reduce header verbosity
            '-p', 'no:cacheprovider'  # Disable cache to avoid conflicts
        ], plugins=[results_capture])
        
        print(f"    Pytest execution completed for {test_file_name}")
        
        if results_capture.results:
            return results_capture.results
        else:
            # Fallback: create basic results structure
            print(f"    No results captured for {test_file_name}, creating basic report")
            return _create_basic_results_structure(test_file_name, exit_code)
            
    except Exception as e:
        print(f"    Error running {test_file_name}: {e}")
        return _create_error_results_structure(test_file_name, str(e))


def _create_basic_results_structure(test_file_name: str, exit_code: int) -> Dict[str, Any]:
    """Create basic results structure when no results are captured"""
    return {
        'test_info': {
            'test_name': test_file_name,
            'timestamp': datetime.now().isoformat(),
            'start_time': time.time(),
            'end_time': time.time(),
            'status': 'completed' if exit_code == 0 else 'failed'
        },
        'summary': {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'errors': 1 if exit_code != 0 else 0,
            'skipped': 0,
            'execution_time': 0.0
        },
        'categories': {},
        'detailed_results': []
    }


def _create_error_results_structure(test_file_name: str, error_message: str) -> Dict[str, Any]:
    """Create error results structure when test execution fails"""
    return {
        'test_info': {
            'test_name': test_file_name,
            'timestamp': datetime.now().isoformat(),
            'start_time': time.time(),
            'end_time': time.time(),
            'status': 'error'
        },
        'summary': {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'errors': 1,
            'skipped': 0,
            'execution_time': 0.0
        },
        'categories': {},
        'detailed_results': [{
            'test_id': f'{test_file_name}_error',
            'status': 'error',
            'category': 'general',
            'error_message': error_message,
            'details': {}
        }]
    }


def _create_basic_results_from_pytest_output(result, test_file_name: str) -> Dict[str, Any]:
    """Create basic results structure from pytest output when no JSON file is available"""
    
    output_lines = result.stdout.split('\n')
    
    # Extract test results from pytest output
    test_results = []
    passed_count = 0
    failed_count = 0
    error_count = 0
    skipped_count = 0
    
    for line in output_lines:
        if '::' in line and ('PASSED' in line or 'FAILED' in line or 'ERROR' in line or 'SKIPPED' in line):
            parts = line.split('::')
            if len(parts) >= 2:
                test_name = parts[-1].split()[0]
                if 'PASSED' in line:
                    status = 'passed'
                    passed_count += 1
                elif 'FAILED' in line:
                    status = 'failed'
                    failed_count += 1
                elif 'ERROR' in line:
                    status = 'error'
                    error_count += 1
                elif 'SKIPPED' in line:
                    status = 'skipped'
                    skipped_count += 1
                else:
                    status = 'unknown'
                
                test_results.append({
                    'test_id': test_name,
                    'status': status,
                    'category': 'general',
                    'details': {}
                })
    
    # Create a basic results structure
    return {
        'test_info': {
            'test_name': test_file_name,
            'timestamp': datetime.now().isoformat(),
            'start_time': time.time(),
            'end_time': time.time(),
            'status': 'completed' if result.returncode == 0 else 'failed'
        },
        'summary': {
            'total_tests': passed_count + failed_count + error_count + skipped_count,
            'passed': passed_count,
            'failed': failed_count,
            'errors': error_count,
            'skipped': skipped_count,
            'execution_time': 0.0
        },
        'categories': {
            'general': {
                'total': passed_count + failed_count + error_count + skipped_count,
                'passed': passed_count,
                'failed': failed_count,
                'errors': error_count,
                'skipped': skipped_count
            }
        },
        'detailed_results': test_results
    }
