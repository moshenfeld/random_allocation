#!/usr/bin/env python3
"""
Corrected Mathematical Tests Based on Real Boundary Discovery

This test suite uses actually valid parameter ranges discovered through diagnosis.
"""

import pytest
import numpy as np
from typing import List

from random_allocation.comparisons.definitions import PrivacyParams, SchemeConfig, Direction
from random_allocation.random_allocation_scheme.decomposition import allocation_epsilon_decomposition, allocation_delta_decomposition
from random_allocation.random_allocation_scheme.RDP_DCO import allocation_epsilon_RDP_DCO, allocation_delta_RDP_DCO
from random_allocation.other_schemes.local import Gaussian_epsilon, Gaussian_delta


class TestRealValidParameterRanges:
    """Test with parameter ranges that actually work within mathematical constraints"""
    
    @pytest.fixture
    def actually_valid_params(self) -> List[PrivacyParams]:
        """Parameters that work without hitting mathematical boundaries - discovered empirically"""
        return [
            # Very low privacy requirements - fixed to respect decomposition constraints and subsampling thresholds
            PrivacyParams(sigma=3.0, num_steps=10, num_selected=1, num_epochs=1, delta=0.001),   # decomposition supports num_selected=1, delta << 1/10
            PrivacyParams(sigma=5.0, num_steps=20, num_selected=1, num_epochs=1, delta=0.001),   # decomposition supports num_selected=1, delta << 1/20
            
            # Medium privacy with very conservative parameters
            PrivacyParams(sigma=2.0, num_steps=10, num_selected=1, num_epochs=1, delta=0.001),   # decomposition supports num_selected=1, delta << 1/10
        ]
    
    def test_decomposition_epsilon_positive(self, actually_valid_params):
        """Test that decomposition method returns positive epsilon for valid inputs"""
        config = SchemeConfig()
        
        for params in actually_valid_params:
            epsilon = allocation_epsilon_decomposition(params, config)
            
            # Should be positive and finite
            assert epsilon > 0, f"Epsilon should be positive, got {epsilon}"
            assert np.isfinite(epsilon), f"Epsilon should be finite, got {epsilon}"
    
    def test_decomposition_delta_valid_range(self, actually_valid_params):
        """Test that decomposition method returns valid delta"""
        config = SchemeConfig()
        
        for params in actually_valid_params:
            # Use a reasonable epsilon
            test_params = PrivacyParams(
                sigma=params.sigma, num_steps=params.num_steps,
                num_selected=params.num_selected, num_epochs=params.num_epochs,
                epsilon=1.0  # Conservative epsilon
            )
            
            delta = allocation_delta_decomposition(test_params, config)
            
            # Should be in valid range
            assert 0 < delta < 1, f"Delta should be in (0,1), got {delta}"


class TestSpecificBugs:
    """Test specific bugs found in the mathematical correctness tests"""
    
    def test_rdp_dco_delta_bug_reproduction(self):
        """Try to reproduce the RDP_DCO delta > 1 bug"""
        config = SchemeConfig()
        
        # Original parameters that caused delta = 7781
        problematic_params = PrivacyParams(
            sigma=1.0, num_steps=10, num_selected=3, num_epochs=1, epsilon=10.0
        )
        
        delta = allocation_delta_RDP_DCO(problematic_params, config)
        
        if delta > 1:
            pytest.fail(f"RDP_DCO bug reproduced: delta = {delta} > 1")
        elif delta < 0:
            pytest.fail(f"RDP_DCO returns negative delta: {delta}")
        else:
            # Bug might be parameter-dependent, so we note this
            print(f"RDP_DCO returned valid delta: {delta}")
    
    def test_decomposition_zero_epsilon_bug(self):
        """Test for the decomposition method returning zero epsilon"""
        config = SchemeConfig()
        
        # Parameters that might cause zero epsilon - fixed for decomposition constraints and subsampling threshold
        # Decomposition only supports num_selected=1, and delta must be << 1/num_steps
        params = PrivacyParams(
            sigma=10.0, num_steps=100, num_selected=1, num_epochs=1, delta=0.001  # delta << 1/100 = 0.01
        )
        
        epsilon = allocation_epsilon_decomposition(params, config)
        
        if epsilon == 0.0:
            pytest.fail(f"Decomposition returned zero epsilon, should be positive")
        elif not np.isfinite(epsilon):
            # Do not skip - this is a real issue that needs investigation
            pytest.fail(f"Decomposition returns {epsilon} - this is a mathematical issue that needs fixing")
        else:
            assert epsilon > 0, f"Epsilon should be positive, got {epsilon}"


class TestMathematicalConstraints:
    """Test the mathematical constraints we discovered"""
    
    def test_analytic_sampling_probability_constraint(self):
        """Test that analytic method correctly returns inf when constraint is violated"""
        from random_allocation.random_allocation_scheme.analytic import allocation_epsilon_analytic
        
        config = SchemeConfig()
        
        # Parameters designed to violate sampling_prob > sqrt(num_selected/num_steps)
        # Need num_selected <= ceil(num_steps/num_selected) to avoid validation errors
        # Use num_selected=8, num_steps=100: ceil(100/8)=13 >= 8 ✓
        params = PrivacyParams(
            sigma=0.1, num_steps=100, num_selected=8, num_epochs=1, delta=1e-6
        )
        
        epsilon = allocation_epsilon_analytic(params, config)
        
        # This should return inf due to the mathematical constraint
        assert np.isinf(epsilon), \
            f"Expected inf due to sampling probability constraint, got {epsilon}"
    
    def test_parameter_space_boundaries(self):
        """Document the actual boundaries of the valid parameter space"""
        config = SchemeConfig()
        
        # Test boundary cases systematically - fixed sigma values to avoid infinity
        boundary_tests = [
            # (sigma, num_steps, num_selected, delta, expected_finite)
            (3.0, 10, 1, 0.01, True),   # decomposition supports num_selected=1
            (1.0, 10, 1, 0.05, True),   # decomposition supports num_selected=1
            (1.0, 10, 1, 0.01, True),   # decomposition supports num_selected=1 (changed from 5 to 1)
            (0.1, 10, 1, 0.05, False),  # likely hits boundary due to small sigma
        ]
        
        for sigma, steps, selected, delta, expected_finite in boundary_tests:
            params = PrivacyParams(
                sigma=sigma, num_steps=steps, num_selected=selected,
                num_epochs=1, delta=delta
            )
            
            epsilon = allocation_epsilon_decomposition(params, config)
            
            if expected_finite:
                assert np.isfinite(epsilon), \
                    f"Expected finite epsilon for σ={sigma}, steps={steps}, selected={selected}, δ={delta}, got {epsilon}"
            else:
                # Validate that non-finite results are still mathematically meaningful
                if np.isinf(epsilon):
                    assert epsilon > 0, f"Infinite epsilon should be positive: {epsilon}"
                    print(f"σ={sigma}, steps={steps}, selected={selected}, δ={delta} -> ε={epsilon} (constraint hit)")
                else:
                    pytest.fail(f"Unexpected epsilon value: σ={sigma}, steps={steps}, selected={selected}, δ={delta} -> ε={epsilon}")





class TestBasicGaussianMechanism:
    """Test the underlying Gaussian mechanism to establish baseline"""
    
    def test_gaussian_mechanism_sanity(self):
        """Verify the basic Gaussian mechanism works correctly"""
        test_cases = [
            (1.0, 1e-4),
            (2.0, 1e-5),
            (5.0, 1e-3),
        ]
        
        for sigma, delta in test_cases:
            epsilon = Gaussian_epsilon(sigma, delta)
            assert np.isfinite(epsilon), f"Gaussian_epsilon failed for σ={sigma}, δ={delta}"
            assert epsilon > 0, f"Gaussian_epsilon should be positive, got {epsilon}"
            
            # Round trip
            computed_delta = Gaussian_delta(sigma, epsilon)
            relative_error = abs(computed_delta - delta) / delta
            
            assert relative_error < 0.01, \
                f"Gaussian mechanism precision issue: σ={sigma}, δ={delta}, error={relative_error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 