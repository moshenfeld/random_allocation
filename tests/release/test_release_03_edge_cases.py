#!/usr/bin/env python3
"""
Edge Case Tests - Boundary conditions and extreme parameter values
Tests functions with edge cases like num_steps=1, num_selections=num_steps, etc.
"""
import pytest
import numpy as np
import time
import json
import os
from datetime import datetime
from typing import Optional
from random_allocation.comparisons.structs import PrivacyParams, SchemeConfig, Direction
import inspect
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from test_utils import ResultsReporter

# All schemes and directions - reuse from monotonicity tests
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

# Edge case parameter sets - OPTIMIZED for fast execution (all tests should be <1 second)
EDGE_CASES = [
    # case_name, description, base_params_epsilon, base_params_delta
    ("minimal_steps", "num_steps=1", 
     dict(sigma=2.0, num_steps=1, num_selected=1, num_epochs=1, sampling_probability=1.0, delta=1e-6, epsilon=None),
     dict(sigma=2.0, num_steps=1, num_selected=1, num_epochs=1, sampling_probability=1.0, epsilon=0.4, delta=None)),
    
    ("equal_selection_steps", "num_selected=num_steps", 
     dict(sigma=2.0, num_steps=3, num_selected=3, num_epochs=1, sampling_probability=1.0, delta=1e-6, epsilon=None),
     dict(sigma=2.0, num_steps=3, num_selected=3, num_epochs=1, sampling_probability=1.0, epsilon=0.4, delta=None)),
    
    ("minimal_sigma", "sigma=0.01 (small)", 
     dict(sigma=0.01, num_steps=5, num_selected=1, num_epochs=1, sampling_probability=1.0, delta=1e-6, epsilon=None),
     dict(sigma=0.01, num_steps=5, num_selected=1, num_epochs=1, sampling_probability=1.0, epsilon=0.4, delta=None)),
    
    ("large_sigma", "sigma=100.0 (large)", 
     dict(sigma=100.0, num_steps=5, num_selected=1, num_epochs=1, sampling_probability=1.0, delta=1e-6, epsilon=None),
     dict(sigma=100.0, num_steps=5, num_selected=1, num_epochs=1, sampling_probability=1.0, epsilon=0.4, delta=None)),
    
    ("tiny_delta", "delta=1e-14 (small)", 
     dict(sigma=2.0, num_steps=5, num_selected=1, num_epochs=1, sampling_probability=1.0, delta=1e-14, epsilon=None),
     None),
    
    ("large_delta", "delta=0.99 (relatively large)", 
     dict(sigma=2.0, num_steps=5, num_selected=1, num_epochs=1, sampling_probability=1.0, delta=0.99, epsilon=None),
     None),
    
    ("tiny_epsilon", "epsilon=0.01 (small)", 
     None,
     dict(sigma=2.0, num_steps=5, num_selected=1, num_epochs=1, sampling_probability=1.0, epsilon=0.01, delta=None)),

    ("large_epsilon", "epsilon=100.0 (large)", 
     None,
     dict(sigma=2.0, num_steps=5, num_selected=1, num_epochs=1, sampling_probability=1.0, epsilon=100.0, delta=None)),
    
    ("minimal_sampling_prob", "sampling_probability=0.01 (small)", 
     dict(sigma=2.0, num_steps=10, num_selected=1, num_epochs=1, sampling_probability=0.01, delta=1e-6, epsilon=None),
     dict(sigma=2.0, num_steps=10, num_selected=1, num_epochs=1, sampling_probability=0.01, epsilon=0.4, delta=None)),
]

# Default scheme configuration - OPTIMIZED for fast edge case testing
config = SchemeConfig()
config.allocation_direct_alpha_orders = list(range(2, 11))  # Reduced from 2-51 to 2-11
config.MC_conf_level = 0.70
config.MC_sample_size = 100  # Reduced from 1,000 to 100 for speed

# Global test results reporter
reporter: Optional[ResultsReporter] = None

@pytest.fixture(scope="session", autouse=True)
def setup_reporter():
    """Setup the global test results reporter for the session."""
    global reporter
    reporter = ResultsReporter("test_release_03_edge_cases")
    yield reporter
    # Save results - but only if not running as part of suite
    is_suite_run = os.environ.get('PYTEST_SUITE_RUN', 'false').lower() == 'true'
    
    if is_suite_run:
        # Just finalize results for suite collection
        results = reporter.get_results()
        print(f"\nEdge case test completed - results will be collected by suite runner")
    else:
        # Save individual JSON file when run standalone
        filepath = reporter.finalize_and_save()
        print(f"\nEdge case test results saved to: {filepath}")

def get_reporter() -> ResultsReporter:
    """Get the global reporter, ensuring it's initialized."""
    assert reporter is not None, "Reporter not initialized. This should not happen."
    return reporter

def function_exists(module_path: str, function_name: str) -> bool:
    """Check if a function exists in a module."""
    try:
        import importlib
        module = importlib.import_module(module_path)
        return hasattr(module, function_name) and getattr(module, function_name) is not None
    except ImportError:
        return False

# Approved invalid settings based on actual parameter restrictions for edge cases we test
APPROVED_INVALID = [
    # Schemes that only support num_selected=1
    ('Shuffle',             'epsilon',  'equal_selection_steps'),
    ('Shuffle',             'delta',    'equal_selection_steps'),
    ('Decomposition',       'epsilon',  'equal_selection_steps'),
    ('Decomposition',       'delta',    'equal_selection_steps'),
    ('LowerBound',          'epsilon',  'equal_selection_steps'),
    ('LowerBound',          'delta',    'equal_selection_steps'),
    ('MonteCarloHighProb',  'delta',    'equal_selection_steps'),
    ('MonteCarloMean',      'delta',    'equal_selection_steps'),

    # Schemes that require sampling_probability=1.0 (fail with minimal_sampling_prob)
    ('Local',               'epsilon',  'minimal_sampling_prob'),
    ('Local',               'delta',    'minimal_sampling_prob'),
    ('PoissonPLD',          'epsilon',  'minimal_sampling_prob'),
    ('PoissonPLD',          'delta',    'minimal_sampling_prob'),
    ('PoissonRDP',          'epsilon',  'minimal_sampling_prob'),
    ('PoissonRDP',          'delta',    'minimal_sampling_prob'),
    ('Shuffle',             'epsilon',  'minimal_sampling_prob'),
    ('Shuffle',             'delta',    'minimal_sampling_prob'),
    ('Direct',              'epsilon',  'minimal_sampling_prob'),
    ('Direct',              'delta',    'minimal_sampling_prob'),
    ('LowerBound',          'epsilon',  'minimal_sampling_prob'),
    ('LowerBound',          'delta',    'minimal_sampling_prob'),
    ('MonteCarloHighProb',  'delta',    'minimal_sampling_prob'),
    ('MonteCarloMean',      'delta',    'minimal_sampling_prob'),
    ('RDP_DCO',             'epsilon',  'minimal_sampling_prob'),
    ('RDP_DCO',             'delta',    'minimal_sampling_prob'),
    ('Decomposition',       'epsilon',  'minimal_sampling_prob'),
    ('Decomposition',       'delta',    'minimal_sampling_prob'),
]

# These are approved timeout cases that cause excessive computation time (>10 seconds)
# These are permanently skipped as they represent computational bottlenecks, not bugs
APPROVED_TIMEOUTS = [
    # PoissonPLD with minimal sigma causes timeout in dp_accounting library
    # This affects several schemes that depend on PoissonPLD
    ('PoissonPLD', 'epsilon', 'minimal_sigma'), 
    ('PoissonPLD', 'delta', 'minimal_sigma'),
    ('Combined', 'epsilon', 'minimal_sigma'),
    ('Combined', 'delta', 'minimal_sigma'),
    ('Recursive', 'epsilon', 'minimal_sigma'),
    ('Recursive', 'delta', 'minimal_sigma'),
    ('Decomposition', 'epsilon', 'minimal_sigma'),
    ('Decomposition', 'delta', 'minimal_sigma'),
 ]

# These are documented bugs that will now FAIL the tests (no longer skipped)
# This list serves as documentation of known issues that need to be fixed



# Helper function to call functions with optional sampling_prob (same as monotonicity tests)
def _call(func, params, config, direction):
    """Call function with appropriate arguments based on signature"""
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

def check_edge_case(func, params, config, direction, case_name, scheme_name, test_type):
    """Check if function can handle edge case parameters without errors"""
    test_reporter = get_reporter()
    
    # Check if this edge case is in approved invalid list (these are expected errors)
    if (scheme_name, test_type, case_name) in APPROVED_INVALID:
        test_reporter.add_test_result(
            test_id=f"{scheme_name}_{test_type}_{direction}_{case_name}",
            category="approved_invalid",
            status="skipped",
            details={
                "scheme": scheme_name,
                "test_type": test_type,
                "direction": str(direction),
                "case_name": case_name,
                "reason": "Approved invalid parameters"
            }
        )
        pytest.skip(f"Edge case '{case_name}' approved as invalid for {scheme_name} {test_type}")
    
    # Check if this is an approved timeout case (computational bottleneck)
    if (scheme_name, test_type, case_name) in APPROVED_TIMEOUTS:
        test_reporter.add_test_result(
            test_id=f"{scheme_name}_{test_type}_{direction}_{case_name}",
            category="approved_timeout",
            status="skipped",
            details={
                "scheme": scheme_name,
                "test_type": test_type,
                "direction": str(direction),
                "case_name": case_name,
                "reason": "Approved timeout"
            }
        )
        pytest.skip(f"Edge case '{case_name}' is an approved timeout for {scheme_name} {test_type}")
        
    # Set timeout threshold (in seconds)
    TIMEOUT_THRESHOLD = 10.0
    
    start_time = time.time()
    try:
        # Create params object - this will validate parameters automatically
        try:
            params_obj = PrivacyParams(**params.__dict__ if hasattr(params, '__dict__') else params)
        except ValueError as e:
            # Parameter validation failed - this is a valid skip for invalid parameters
            elapsed_time = time.time() - start_time
            test_reporter.add_test_result(
                test_id=f"{scheme_name}_{test_type}_{direction}_{case_name}",
                category="invalid_params",
                status="skipped",
                details={
                    "scheme": scheme_name,
                    "test_type": test_type,
                    "direction": str(direction),
                    "case_name": case_name,
                    "reason": f"Invalid parameters: {str(e)}"
                },
                execution_time=elapsed_time
            )
            pytest.skip(f"Edge case '{case_name}' has invalid parameters for {scheme_name} {test_type}: {str(e)}")
        
        result = _call(func, params_obj, config, direction)
        elapsed_time = time.time() - start_time
        
        # Check for timeout
        if elapsed_time > TIMEOUT_THRESHOLD:
            raise TimeoutError(f"Function took {elapsed_time:.2f}s, exceeding {TIMEOUT_THRESHOLD}s threshold")
        
        # Check result validity
        if result is None:
            raise ValueError("Function returned None")
        
        if not isinstance(result, (int, float, np.number)):
            raise ValueError(f"Function returned invalid type: {type(result)}")
        
        if np.isnan(result):
            raise ValueError(f"Function returned NaN: {result}")
        
        # Treat inf as valid for edge cases (boundary conditions can legitimately return inf)
        if np.isinf(result):
            print(f"⚠️  {scheme_name}.{test_type} with {direction} returned inf for '{case_name}' (time: {elapsed_time:.2f}s)")
            test_reporter.add_test_result(
                test_id=f"{scheme_name}_{test_type}_{direction}_{case_name}",
                category="edge_case_success",
                status="passed",
                details={
                    "scheme": scheme_name,
                    "test_type": test_type,
                    "direction": str(direction),
                    "case_name": case_name,
                    "result": "inf",
                    "reason": "Returned inf (valid for edge cases)"
                },
                execution_time=elapsed_time
            )
            return result  # inf is valid for edge cases
        
        if result < 0:
            raise ValueError(f"Function returned negative value: {result}")
        
        # For delta, check it's <= 1 (but inf is still allowed)
        if test_type == "delta" and result > 1.0:
            raise ValueError(f"Delta value exceeds 1.0: {result}")
        
        # Log timing for successful cases
        test_reporter.add_test_result(
            test_id=f"{scheme_name}_{test_type}_{direction}_{case_name}",
            category="edge_case_success",
            status="passed",
            details={
                "scheme": scheme_name,
                "test_type": test_type,
                "direction": str(direction),
                "case_name": case_name,
                "result": float(result) if not np.isinf(result) else "inf",
                "reason": "Success"
            },
            execution_time=elapsed_time
        )
        
        if elapsed_time > 1.0:  # Log slow but successful cases
            print(f"🐌 {scheme_name}.{test_type} with {direction} completed '{case_name}' slowly (time: {elapsed_time:.2f}s): {result}")
        elif elapsed_time > 0.5:
            print(f"⏱️  {scheme_name}.{test_type} with {direction} completed '{case_name}' (time: {elapsed_time:.2f}s): {result}")
        
        return result
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        
        # Log the timing info for failed cases
        print(f"❌ {scheme_name}.{test_type} with {direction} failed '{case_name}' after {elapsed_time:.2f}s: {str(e)}")
        
        # Store failure information
        test_reporter.add_test_result(
            test_id=f"{scheme_name}_{test_type}_{direction}_{case_name}",
            category="edge_case_failure",
            status="failed",
            details={
                "scheme": scheme_name,
                "test_type": test_type,
                "direction": str(direction),
                "case_name": case_name,
                "reason": str(e)
            },
            error_message=str(e),
            execution_time=elapsed_time
        )
        
        # Categorize timeouts
        if isinstance(e, TimeoutError) or elapsed_time > TIMEOUT_THRESHOLD:
            print(f"💡 Timeout after {elapsed_time:.2f}s for {scheme_name} {test_type} {case_name}")
        
        # Unapproved failure - this is a real issue that needs attention
        pytest.fail(f"Edge case '{case_name}' failed for {scheme_name}.{test_type} with {direction} after {elapsed_time:.2f}s: {str(e)}")

class TestEdgeCases:
    """Test all schemes with edge case parameters"""
    
    @pytest.mark.parametrize("scheme_name,direction,eps_name,delta_name,module_path,case_name,description,eps_params,delta_params", 
                              [(scheme[0], scheme[1], scheme[2], scheme[3], scheme[4], case[0], case[1], case[2], case[3])
                               for scheme in SCHEMES for case in EDGE_CASES 
                               if scheme[2] is not None and case[2] is not None and function_exists(scheme[4], scheme[2])])  # Epsilon tests: need eps_name, eps_params, and function must exist
    def test_epsilon_edge_cases(self, scheme_name, direction, eps_name, delta_name, module_path, 
                              case_name, description, eps_params, delta_params):
        """Test epsilon functions with edge case parameters"""
        test_reporter = get_reporter()
        
        # Import module and get function
        try:
            import importlib
            module = importlib.import_module(module_path)
            func = getattr(module, eps_name)
        except ImportError:
            test_reporter.add_test_result(
                test_id=f"{scheme_name}_epsilon_{direction}_{case_name}",
                category="import_error",
                status="error",
                details={
                    "scheme": scheme_name,
                    "test_type": "epsilon",
                    "direction": str(direction),
                    "case_name": case_name,
                    "reason": f"Cannot import module {module_path}"
                },
                error_message=f"Import error: {module_path}"
            )
            pytest.fail(f"Cannot import module {module_path}")
        
        # Test the edge case
        check_edge_case(func, PrivacyParams(**eps_params), config, direction, case_name, scheme_name, "epsilon")
    
    @pytest.mark.parametrize("scheme_name,direction,eps_name,delta_name,module_path,case_name,description,eps_params,delta_params", 
                              [(scheme[0], scheme[1], scheme[2], scheme[3], scheme[4], case[0], case[1], case[2], case[3])
                               for scheme in SCHEMES for case in EDGE_CASES 
                               if scheme[3] is not None and case[3] is not None and function_exists(scheme[4], scheme[3])])  # Delta tests: need delta_name, delta_params, and function must exist
    def test_delta_edge_cases(self, scheme_name, direction, eps_name, delta_name, module_path,
                            case_name, description, eps_params, delta_params):
        """Test delta functions with edge case parameters"""
        test_reporter = get_reporter()
        
        # Import module and get function
        try:
            import importlib
            module = importlib.import_module(module_path)
            func = getattr(module, delta_name)
        except ImportError:
            test_reporter.add_test_result(
                test_id=f"{scheme_name}_delta_{direction}_{case_name}",
                category="import_error",
                status="error",
                details={
                    "scheme": scheme_name,
                    "test_type": "delta",
                    "direction": str(direction),
                    "case_name": case_name,
                    "reason": f"Cannot import module {module_path}"
                },
                error_message=f"Import error: {module_path}"
            )
            pytest.fail(f"Cannot import module {module_path}")
        
        # Test the edge case
        check_edge_case(func, PrivacyParams(**delta_params), config, direction, case_name, scheme_name, "delta")
