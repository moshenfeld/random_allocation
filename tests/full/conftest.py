#!/usr/bin/env python3
"""
Pytest configuration for full tests - captures pytest results for suite reporting
"""
import pytest
import os
from tests.test_utils import ResultsReporter


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
            
            # Add test result
            reporter.add_test_result(
                test_id=item.name,
                category="general",
                status=status,
                details={},
                error_message=str(rep.longrepr) if rep.failed else None
            )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(session, config, items):
    """Initialize the reporter after items are collected"""
    # Always initialize reporter - the output directory will be controlled by environment
    if items:
        # Extract test file name from first test item
        first_item = items[0]
        test_file_name = first_item.fspath.basename.replace('.py', '')
        
        output_dir = os.environ.get('PYTEST_TEST_RESULTS_DIR', 'tests/test_results')
        session._capture_reporter = ResultsReporter(test_file_name, output_dir)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Save results at the end of the session (only when not running as part of suite)"""
    is_suite_run = os.environ.get('PYTEST_SUITE_RUN', 'false').lower() == 'true'
    if hasattr(session, '_capture_reporter'):
        if not is_suite_run:
            # Only save to JSON file when running standalone
            session._capture_reporter.finalize_and_save()
        else:
            # For suite runs, just finalize the results in memory
            session._capture_reporter.get_results()
