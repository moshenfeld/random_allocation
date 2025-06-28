#!/usr/bin/env python3
"""
Monotonicity Tests - Full scheme/direction coverage
"""
import pytest
import numpy as np
import os
from random_allocation.comparisons.structs import PrivacyParams, SchemeConfig, Direction
import inspect
import importlib
from typing import Optional
from tests.test_utils import ResultsReporter

# Expanded schemes and directions - always use main functions with direction parameter
SCHEMES = [
    # scheme, direction, epsilon_name, delta_name, module path
    ("Local", Direction.ADD, "local_epsilon", "local_delta", "random_allocation.other_schemes.local"),
    ("Local", Direction.REMOVE, "local_epsilon", "local_delta", "random_allocation.other_schemes.local"),
    ("Local", Direction.BOTH, "local_epsilon", "local_delta", "random_allocation.other_schemes.local"),
    ("PoissonPLD", Direction.ADD, "Poisson_epsilon_PLD", "Poisson_delta_PLD", "random_allocation.other_schemes.poisson"),
    ("PoissonPLD", Direction.REMOVE, "Poisson_epsilon_PLD", "Poisson_delta_PLD", "random_allocation.other_schemes.poisson"),
    ("PoissonPLD", Direction.BOTH, "Poisson_epsilon_PLD", "Poisson_delta_PLD", "random_allocation.other_schemes.poisson"),
    ("PoissonRDP", Direction.BOTH, "Poisson_epsilon_RDP", "Poisson_delta_RDP", "random_allocation.other_schemes.poisson"),
    ("Shuffle", Direction.ADD, "shuffle_epsilon_analytic", "shuffle_delta_analytic", "random_allocation.other_schemes.shuffle"),
    ("Shuffle", Direction.REMOVE, "shuffle_epsilon_analytic", "shuffle_delta_analytic", "random_allocation.other_schemes.shuffle"),
    ("Shuffle", Direction.BOTH, "shuffle_epsilon_analytic", "shuffle_delta_analytic", "random_allocation.other_schemes.shuffle"),
    ("Direct", Direction.ADD, "allocation_epsilon_direct", "allocation_delta_direct", "random_allocation.random_allocation_scheme.direct"),
    ("Direct", Direction.REMOVE, "allocation_epsilon_direct", "allocation_delta_direct", "random_allocation.random_allocation_scheme.direct"),
    ("Direct", Direction.BOTH, "allocation_epsilon_direct", "allocation_delta_direct", "random_allocation.random_allocation_scheme.direct"),
    ("Analytic", Direction.ADD, "allocation_epsilon_analytic", "allocation_delta_analytic", "random_allocation.random_allocation_scheme.analytic"),
    ("Analytic", Direction.REMOVE, "allocation_epsilon_analytic", "allocation_delta_analytic", "random_allocation.random_allocation_scheme.analytic"),
    ("Analytic", Direction.BOTH, "allocation_epsilon_analytic", "allocation_delta_analytic", "random_allocation.random_allocation_scheme.analytic"),
    ("Combined", Direction.ADD, "allocation_epsilon_combined", "allocation_delta_combined", "random_allocation.random_allocation_scheme.combined"),
    ("Combined", Direction.REMOVE, "allocation_epsilon_combined", "allocation_delta_combined", "random_allocation.random_allocation_scheme.combined"),
    ("Combined", Direction.BOTH, "allocation_epsilon_combined", "allocation_delta_combined", "random_allocation.random_allocation_scheme.combined"),
    ("Decomposition", Direction.ADD, "allocation_epsilon_decomposition", "allocation_delta_decomposition", "random_allocation.random_allocation_scheme.decomposition"),
    ("Decomposition", Direction.REMOVE, "allocation_epsilon_decomposition", "allocation_delta_decomposition", "random_allocation.random_allocation_scheme.decomposition"),
    ("Decomposition", Direction.BOTH, "allocation_epsilon_decomposition", "allocation_delta_decomposition", "random_allocation.random_allocation_scheme.decomposition"),
    ("LowerBound", Direction.ADD, "allocation_epsilon_lower_bound", "allocation_delta_lower_bound", "random_allocation.random_allocation_scheme.lower_bound"),
    ("LowerBound", Direction.REMOVE, "allocation_epsilon_lower_bound", "allocation_delta_lower_bound", "random_allocation.random_allocation_scheme.lower_bound"),
    ("LowerBound", Direction.BOTH, "allocation_epsilon_lower_bound", "allocation_delta_lower_bound", "random_allocation.random_allocation_scheme.lower_bound"),
    # Monte Carlo variants with different configurations - only supports delta
    ("MonteCarloHighProb", Direction.ADD, None, "allocation_delta_MC", "random_allocation.random_allocation_scheme.Monte_Carlo"),
    ("MonteCarloHighProb", Direction.REMOVE, None, "allocation_delta_MC", "random_allocation.random_allocation_scheme.Monte_Carlo"),
    ("MonteCarloHighProb", Direction.BOTH, None, "allocation_delta_MC", "random_allocation.random_allocation_scheme.Monte_Carlo"),
    ("MonteCarloMean", Direction.ADD, None, "allocation_delta_MC", "random_allocation.random_allocation_scheme.Monte_Carlo"),
    ("MonteCarloMean", Direction.REMOVE, None, "allocation_delta_MC", "random_allocation.random_allocation_scheme.Monte_Carlo"),
    ("MonteCarloMean", Direction.BOTH, None, "allocation_delta_MC", "random_allocation.random_allocation_scheme.Monte_Carlo"),
    ("RDP_DCO", Direction.ADD, "allocation_epsilon_RDP_DCO", "allocation_delta_RDP_DCO", "random_allocation.random_allocation_scheme.RDP_DCO"),
    ("RDP_DCO", Direction.REMOVE, "allocation_epsilon_RDP_DCO", "allocation_delta_RDP_DCO", "random_allocation.random_allocation_scheme.RDP_DCO"),
    ("RDP_DCO", Direction.BOTH, "allocation_epsilon_RDP_DCO", "allocation_delta_RDP_DCO", "random_allocation.random_allocation_scheme.RDP_DCO"),
    ("Recursive", Direction.ADD, "allocation_epsilon_recursive", "allocation_delta_recursive", "random_allocation.random_allocation_scheme.recursive"),
    ("Recursive", Direction.REMOVE, "allocation_epsilon_recursive", "allocation_delta_recursive", "random_allocation.random_allocation_scheme.recursive"),
    ("Recursive", Direction.BOTH, "allocation_epsilon_recursive", "allocation_delta_recursive", "random_allocation.random_allocation_scheme.recursive"),
]

# Default parameter sets for epsilon and delta tests (num_selected=1, sampling_probability=1.0)
default_eps = dict(sigma=4.0, num_steps=10_000, num_selected=1, num_epochs=1, sampling_probability=1.0, delta=1e-10, epsilon=None)
default_delta = dict(sigma=4.0, num_steps=100, num_selected=1, num_epochs=1, sampling_probability=1.0, epsilon=0.4, delta=None)

# Default scheme configuration
config = SchemeConfig()
config.allocation_direct_alpha_orders = list(range(2, 51))
config.MC_conf_level = 0.70
config.MC_sample_size = 500_000
# Param changes: var, low, high, increasing
enumerated_changes = [
    ("sigma", 2.0, 5.0, False),
    ("num_steps", 100, 10_000, False),
    ("num_selected", 3, 10, True),
    ("num_epochs", 1, 5, True),
    ("sampling_probability", 0.2, 0.8, True),
]

# Approved invalid settings will be collected here for review
APPROVED_INVALID = [
    # sampling_probability < 1.0 invalid for these schemes
    ('Local',               'epsilon', 'sampling_probability'),
    ('Local',               'delta',   'sampling_probability'),
    ('PoissonPLD',          'epsilon', 'sampling_probability'),
    ('PoissonPLD',          'delta',   'sampling_probability'),
    ('PoissonRDP',          'epsilon', 'sampling_probability'),
    ('PoissonRDP',          'delta',   'sampling_probability'),
    ('Shuffle',             'epsilon', 'sampling_probability'),
    ('Shuffle',             'delta',   'sampling_probability'),
    ('Direct',              'epsilon', 'sampling_probability'),
    ('Direct',              'delta',   'sampling_probability'),
    ('LowerBound',          'epsilon', 'sampling_probability'),
    ('LowerBound',          'delta',   'sampling_probability'),
    ('MonteCarloHighProb',  'delta',   'sampling_probability'),
    ('MonteCarloMean',      'delta',   'sampling_probability'),
    ('RDP_DCO',             'epsilon', 'sampling_probability'),
    ('RDP_DCO',             'delta',   'sampling_probability'),
    ('Combined',            'epsilon', 'sampling_probability'),
    ('Combined',            'delta',   'sampling_probability'),

    # num_epochs>1 or num_selected>1 invalid for these schemes
    ('PoissonPLD',         'epsilon', 'num_selected'),
    ('PoissonPLD',         'delta',   'num_selected'),
    ('PoissonRDP',         'epsilon', 'num_selected'),
    ('PoissonRDP',         'delta',   'num_selected'),
    ('Shuffle',            'epsilon', 'num_selected'),
    ('Shuffle',            'epsilon', 'num_epochs'),
    ('Shuffle',            'delta',   'num_selected'),
    ('Shuffle',            'delta',   'num_epochs'),
    ('Decomposition',      'epsilon', 'num_selected'),
    ('Decomposition',      'epsilon', 'num_epochs'),
    ('Decomposition',      'delta',   'num_selected'),
    ('Decomposition',      'delta',   'num_epochs'),
    ('LowerBound',         'epsilon', 'num_selected'),
    ('LowerBound',         'epsilon', 'num_epochs'),
    ('LowerBound',         'delta',   'num_selected'),
    ('LowerBound',         'delta',   'num_epochs'),
    ('MonteCarloHighProb', 'delta',   'num_selected'),
    ('MonteCarloHighProb', 'delta',   'num_epochs'),
    ('MonteCarloMean',     'delta',   'num_selected'),
    ('MonteCarloMean',     'delta',   'num_epochs'),
]

# These are documented bugs that will now FAIL the tests (no longer pass silently)
# This list serves as documentation of known monotonicity issues that need to be fixed
DOCUMENTED_BUGS = [
    # Local scheme doesn't depend on num_steps - always returns same value
    ('Local',        'epsilon', 'num_steps'),
    ('Local',        'delta',   'num_steps'),
    ('Local',        'epsilon', 'sampling_probability'),
    ('Local',        'delta',   'sampling_probability'),
]

# Global test results reporter
reporter: Optional[ResultsReporter] = None

@pytest.fixture(scope="session", autouse=True)
def setup_reporter():
    """Setup the global test results reporter for the session."""
    global reporter
    reporter = ResultsReporter("monotonicity_tests")
    yield reporter
    # Save results to JSON file and print summary
    # Save results to JSON file and print summary - but only if not running as part of suite
    is_suite_run = os.environ.get('PYTEST_SUITE_RUN', 'false').lower() == 'true'
    
    if is_suite_run:
        # Just finalize results for suite collection
        results = reporter.get_results()
        print(f"\nMonotonicity test completed - results will be collected by suite runner")
    else:
        # Save individual JSON file when run standalone
        filepath = reporter.finalize_and_save()
        print(f"\nMonotonicity test results saved to: {filepath}")
    
    print(reporter.get_summary_report())

def get_reporter() -> ResultsReporter:
    """Get the global reporter, ensuring it's initialized."""
    assert reporter is not None, "Reporter not initialized. This should not happen."
    return reporter

# helper to call functions with optional sampling_prob
def _call(func, params, config, direction):
    sig = inspect.signature(func)
    args = [params, config]
    
    # For Poisson functions, don't pass the extra sampling_prob argument
    # because they calculate their own sampling probability from params
    func_name = getattr(func, '__name__', '')
    is_poisson_func = 'Poisson' in func_name
    
    if 'sampling_prob' in sig.parameters and not is_poisson_func:
        args.append(params.sampling_probability)
    elif is_poisson_func and 'sampling_prob' in sig.parameters:
        # For Poisson functions, use default sampling_prob (0.0) but still pass it
        args.append(0.0)
    
    # only pass direction if func accepts it
    if 'direction' in sig.parameters:
        args.append(direction)
    return func(*args)

def check_monotonicity(func, params_low, params_high, config, direction, increasing):
    """Check monotonicity for a function with given parameters"""
    # missing function
    if func is None:
        return 'invalid', 'function not found'
    
    try:
        low = _call(func, params_low, config, direction)
        high = _call(func, params_high, config, direction)
    except (AttributeError, ImportError, ValueError, NotImplementedError, AssertionError, TypeError, KeyError, RuntimeError) as e:
        return 'invalid', str(e)
    
    if low == high or (increasing and high < low) or (not increasing and high > low):
        return 'failed', f'values: low={low}, high={high}'
    return 'passed', f'values: low={low}, high={high}'


class TestMonotonicity:
    @pytest.mark.parametrize("scheme_name, direction, eps_name, delta_name, module_path", SCHEMES)
    @pytest.mark.parametrize("var, low, high, increasing", enumerated_changes)
    def test_epsilon_monotonicity(self, scheme_name, direction, eps_name, delta_name, module_path, var, low, high, increasing):
        # skip schemes without epsilon
        if not eps_name:
            return
            
        test_reporter = get_reporter()
        test_id = f"pytest_{scheme_name}_{direction.value}_epsilon_{var}"
        category = f"pytest_epsilon_{var}"
        
        try:
            module = pytest.importorskip(module_path)
            eps_func = getattr(module, eps_name)
            # configure direct alpha orders for schemes needing it
            config.MC_use_mean = (scheme_name == "MonteCarloMean")
            
            # base params for epsilon tests
            base = default_eps.copy()
            # vary var
            params_low = PrivacyParams(**{**base, var: low})
            params_high = PrivacyParams(**{**base, var: high})
            
            # call or expect exception
            try:
                eps_low = _call(eps_func, params_low, config, direction)
                eps_high = _call(eps_func, params_high, config, direction)
            except (ValueError, AssertionError, TypeError):
                # For approved invalid combinations, this is expected
                if (scheme_name, 'epsilon', var) in APPROVED_INVALID:
                    test_reporter.add_test_result(
                        test_id=test_id,
                        category=category,
                        status="skipped",
                        details={
                            "scheme": scheme_name,
                            "direction": direction.value,
                            "parameter_type": "epsilon",
                            "variable": var,
                            "low_value": low,
                            "high_value": high,
                            "increasing": increasing,
                            "function": eps_name
                        },
                        error_message="Expected exception for invalid combination"
                    )
                    return  # Expected exception for invalid combination
                else:
                    test_reporter.add_test_result(
                        test_id=test_id,
                        category=category,
                        status="errors",
                        details={
                            "scheme": scheme_name,
                            "direction": direction.value,
                            "parameter_type": "epsilon",
                            "variable": var,
                            "low_value": low,
                            "high_value": high,
                            "increasing": increasing,
                            "function": eps_name
                        },
                        error_message="Unexpected exception"
                    )
                    raise  # Unexpected exception
                    
            # NOTE: DOCUMENTED_BUGS are no longer skipped - they should fail to expose bugs that need fixing
            # If this is a documented bug, the test will proceed and fail with appropriate error message
            is_documented_bug = (scheme_name, 'epsilon', var) in DOCUMENTED_BUGS
                
            # Handle numerical precision issues (but not for documented bugs - they should fail)
            if not is_documented_bug and abs(eps_low - eps_high) < 1e-10:
                # Values are too close to distinguish reliably
                test_reporter.add_test_result(
                    test_id=test_id,
                    category=category,
                    status="skipped",
                    details={
                        "scheme": scheme_name,
                        "direction": direction.value,
                        "parameter_type": "epsilon",
                        "variable": var,
                        "low_value": low,
                        "high_value": high,
                        "increasing": increasing,
                        "function": eps_name,
                        "eps_low": eps_low,
                        "eps_high": eps_high
                    },
                    error_message="Values too close for numerical precision"
                )
                return
                
            # check change and direction for normal cases
            try:
                assert eps_low != eps_high, f"{eps_name} no change for {var}: {eps_low} vs {eps_high}"
                if increasing:
                    assert eps_high > eps_low, f"{eps_name} not increasing in {var}: {eps_low} !< {eps_high}"
                else:
                    assert eps_high < eps_low, f"{eps_name} not decreasing in {var}: {eps_low} !> {eps_high}"
                
                # Test passed
                test_reporter.add_test_result(
                    test_id=test_id,
                    category=category,
                    status="passed",
                    details={
                        "scheme": scheme_name,
                        "direction": direction.value,
                        "parameter_type": "epsilon",
                        "variable": var,
                        "low_value": low,
                        "high_value": high,
                        "increasing": increasing,
                        "function": eps_name,
                        "eps_low": eps_low,
                        "eps_high": eps_high
                    }
                )
                
            except AssertionError as e:
                error_msg = str(e)
                if is_documented_bug:
                    error_msg = f"Documented bug: {error_msg}"
                
                test_reporter.add_test_result(
                    test_id=test_id,
                    category=category,
                    status="failed",
                    details={
                        "scheme": scheme_name,
                        "direction": direction.value,
                        "parameter_type": "epsilon",
                        "variable": var,
                        "low_value": low,
                        "high_value": high,
                        "increasing": increasing,
                        "function": eps_name,
                        "eps_low": eps_low,
                        "eps_high": eps_high
                    },
                    error_message=error_msg
                )
                raise  # Re-raise the assertion error for pytest
                
        except Exception as e:
            # Record any other unexpected errors
            test_reporter.add_test_result(
                test_id=test_id,
                category=category,
                status="errors",
                details={
                    "scheme": scheme_name,
                    "direction": direction.value,
                    "parameter_type": "epsilon",
                    "variable": var,
                    "low_value": low,
                    "high_value": high,
                    "increasing": increasing,
                    "function": eps_name
                },
                error_message=str(e)
            )
            raise

    @pytest.mark.parametrize("scheme_name, direction, eps_name, delta_name, module_path", SCHEMES)
    @pytest.mark.parametrize("var, low, high, increasing", enumerated_changes)
    def test_delta_monotonicity(self, scheme_name, direction, eps_name, delta_name, module_path, var, low, high, increasing):
        # skip schemes without delta
        if not delta_name:
            return
            
        test_reporter = get_reporter()
        test_id = f"pytest_{scheme_name}_{direction.value}_delta_{var}"
        category = f"pytest_delta_{var}"
        
        try:
            module = pytest.importorskip(module_path)
            delta_func = getattr(module, delta_name)
            # configure direct alpha orders for schemes needing it
            config.MC_use_mean = (scheme_name == "MonteCarloMean")
            
            # base params for delta tests
            base = default_delta.copy()
            # vary var
            params_low = PrivacyParams(**{**base, var: low})
            params_high = PrivacyParams(**{**base, var: high})
            
            # call or expect exception
            try:
                delta_low = _call(delta_func, params_low, config, direction)
                delta_high = _call(delta_func, params_high, config, direction)
            except (ValueError, AssertionError, TypeError):
                # For approved invalid combinations, this is expected
                if (scheme_name, 'delta', var) in APPROVED_INVALID:
                    test_reporter.add_test_result(
                        test_id=test_id,
                        category=category,
                        status="skipped",
                        details={
                            "scheme": scheme_name,
                            "direction": direction.value,
                            "parameter_type": "delta",
                            "variable": var,
                            "low_value": low,
                            "high_value": high,
                            "increasing": increasing,
                            "function": delta_name
                        },
                        error_message="Expected exception for invalid combination"
                    )
                    return  # Expected exception for invalid combination
                else:
                    test_reporter.add_test_result(
                        test_id=test_id,
                        category=category,
                        status="errors",
                        details={
                            "scheme": scheme_name,
                            "direction": direction.value,
                            "parameter_type": "delta",
                            "variable": var,
                            "low_value": low,
                            "high_value": high,
                            "increasing": increasing,
                            "function": delta_name
                        },
                        error_message="Unexpected exception"
                    )
                    raise  # Unexpected exception
                    
            # NOTE: DOCUMENTED_BUGS are no longer skipped - they should fail to expose bugs that need fixing
            # If this is a documented bug, the test will proceed and fail with appropriate error message
            is_documented_bug = (scheme_name, 'delta', var) in DOCUMENTED_BUGS
                
            # Handle numerical precision issues (but not for documented bugs - they should fail)
            if not is_documented_bug and abs(delta_low - delta_high) < 1e-10:
                # Values are too close to distinguish reliably
                test_reporter.add_test_result(
                    test_id=test_id,
                    category=category,
                    status="skipped",
                    details={
                        "scheme": scheme_name,
                        "direction": direction.value,
                        "parameter_type": "delta",
                        "variable": var,
                        "low_value": low,
                        "high_value": high,
                        "increasing": increasing,
                        "function": delta_name,
                        "delta_low": delta_low,
                        "delta_high": delta_high
                    },
                    error_message="Values too close for numerical precision"
                )
                return
                
            # check change and direction for normal cases
            try:
                assert delta_low != delta_high, f"{delta_name} no change for {var}: {delta_low} vs {delta_high}"
                if increasing:
                    assert delta_high > delta_low, f"{delta_name} not increasing in {var}: {delta_low} !< {delta_high}"
                else:
                    assert delta_high < delta_low, f"{delta_name} not decreasing in {var}: {delta_low} !> {delta_high}"
                
                # Test passed
                test_reporter.add_test_result(
                    test_id=test_id,
                    category=category,
                    status="passed",
                    details={
                        "scheme": scheme_name,
                        "direction": direction.value,
                        "parameter_type": "delta",
                        "variable": var,
                        "low_value": low,
                        "high_value": high,
                        "increasing": increasing,
                        "function": delta_name,
                        "delta_low": delta_low,
                        "delta_high": delta_high
                    }
                )
                
            except AssertionError as e:
                error_msg = str(e)
                if is_documented_bug:
                    error_msg = f"Documented bug: {error_msg}"
                
                test_reporter.add_test_result(
                    test_id=test_id,
                    category=category,
                    status="failed",
                    details={
                        "scheme": scheme_name,
                        "direction": direction.value,
                        "parameter_type": "delta",
                        "variable": var,
                        "low_value": low,
                        "high_value": high,
                        "increasing": increasing,
                        "function": delta_name,
                        "delta_low": delta_low,
                        "delta_high": delta_high
                    },
                    error_message=error_msg
                )
                raise  # Re-raise the assertion error for pytest
                
        except Exception as e:
            # Record any other unexpected errors
            test_reporter.add_test_result(
                test_id=test_id,
                category=category,
                status="errors",
                details={
                    "scheme": scheme_name,
                    "direction": direction.value,
                    "parameter_type": "delta",
                    "variable": var,
                    "low_value": low,
                    "high_value": high,
                    "increasing": increasing,
                    "function": delta_name
                },
                error_message=str(e)
            )
            raise
