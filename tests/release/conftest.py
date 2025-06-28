#!/usr/bin/env python3
"""
Pytest configuration for release tests - adds progress tracking and results capture
"""
import pytest
import time
import os
from tests.test_utils import ResultsReporter


class ProgressTracker:
    def __init__(self):
        self.start_time = time.time()
        self.test_count = 0
        self.total_tests = 0
        self.last_report_time = time.time()
        self.report_interval = 30  # Report every 30 seconds

    def set_total(self, total):
        self.total_tests = total

    def increment(self):
        self.test_count += 1
        current_time = time.time()
        
        # Report progress every 30 seconds or every 50 tests
        if (current_time - self.last_report_time > self.report_interval) or (self.test_count % 50 == 0):
            elapsed = current_time - self.start_time
            if self.total_tests > 0:
                progress_pct = (self.test_count / self.total_tests) * 100
                remaining_tests = self.total_tests - self.test_count
                if self.test_count > 0:
                    avg_time_per_test = elapsed / self.test_count
                    eta_seconds = remaining_tests * avg_time_per_test
                    eta_minutes = eta_seconds / 60
                    print(f"\n  Progress: {self.test_count}/{self.total_tests} tests ({progress_pct:.1f}%) - {elapsed/60:.1f}m elapsed, ~{eta_minutes:.1f}m remaining")
                else:
                    print(f"\n  Progress: {self.test_count}/{self.total_tests} tests ({progress_pct:.1f}%) - {elapsed/60:.1f}m elapsed")
            else:
                print(f"\n  Progress: {self.test_count} tests completed - {elapsed/60:.1f}m elapsed")
            self.last_report_time = current_time


# Global progress tracker
progress_tracker = ProgressTracker()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Capture test results for each test"""
    outcome = yield
    rep = outcome.get_result()
    
    # Only record during the call phase (not setup/teardown)
    if call.when == "call":
        if hasattr(item.session, '_capture_reporter'):
            reporter = item.session._capture_reporter
            
            # Determine status
            if rep.passed:
                status = "passed"
            elif rep.failed:
                status = "failed"
            elif rep.skipped:
                status = "skipped"
            else:
                status = "error"
            
            # Prepare details with skip reason if applicable
            details = {}
            if rep.skipped and hasattr(rep, 'longrepr') and rep.longrepr:
                details['skip_reason'] = str(rep.longrepr)
            
            # Add test result
            reporter.add_test_result(
                test_id=item.name,
                category="general",
                status=status,
                details=details,
                error_message=str(rep.longrepr) if rep.failed else None
            )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Called after collection is completed - set total test count and initialize reporter"""
    progress_tracker.set_total(len(items))
    print(f"\nStarting {len(items)} tests...")
    
    # Initialize the reporter after items are collected
    if items:
        # Extract test file name from first test item
        first_item = items[0]
        test_file_name = first_item.fspath.basename.replace('.py', '')
        
        output_dir = os.environ.get('PYTEST_TEST_RESULTS_DIR', 'tests/test_results')
        first_item.session._capture_reporter = ResultsReporter(test_file_name, output_dir)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    """Called for each test - increment progress counter"""
    yield  # Run the test
    progress_tracker.increment()


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Called when session finishes"""
    elapsed = time.time() - progress_tracker.start_time
    print(f"\nTest session completed: {progress_tracker.test_count} tests in {elapsed/60:.1f} minutes")
    
    # Save results at the end of the session (only when not running as part of suite)
    is_suite_run = os.environ.get('PYTEST_SUITE_RUN', 'false').lower() == 'true'
    if hasattr(session, '_capture_reporter'):
        if not is_suite_run:
            # Only save to JSON file when running standalone
            session._capture_reporter.finalize_and_save()
        else:
            # For suite runs, just finalize the results in memory
            session._capture_reporter.get_results()
