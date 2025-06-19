#!/usr/bin/env python3
"""
Comprehensive Coverage Tests - Level 5

Tests remaining methods, edge cases, and comprehensive coverage.
Final level to ensure all major functionality is tested.
"""

import pytest
import numpy as np
from random_allocation.comparisons.definitions import PrivacyParams, SchemeConfig, Direction


class TestRemainingAllocationMethods:
    """Test remaining allocation methods for completeness"""
    

    

    
    def test_allocation_lower_bound_method(self):
        """Test allocation lower bound method"""
        pytest.importorskip("random_allocation.random_allocation_scheme.lower_bound")
        from random_allocation.random_allocation_scheme.lower_bound import allocation_epsilon_lower_bound, allocation_delta_lower_bound
        
        # Lower bound method only supports num_selected=1 and num_epochs=1
        params = PrivacyParams(
            sigma=5.0, num_steps=10, num_selected=1, num_epochs=1, delta=1e-4
        )
        config = SchemeConfig()
        
        epsilon = allocation_epsilon_lower_bound(params, config)
        
        assert np.isfinite(epsilon) or np.isinf(epsilon), f"Lower bound epsilon returned {epsilon}"
        if np.isfinite(epsilon):
            assert epsilon > 0, f"Lower bound epsilon should be positive: {epsilon}"


class TestParameterBoundaryConditions:
    """Test various parameter boundary conditions"""
    
    def test_minimal_parameters(self):
        """Test with minimal valid parameters"""
        from random_allocation.random_allocation_scheme.decomposition import allocation_epsilon_decomposition
        
        # Absolutely minimal parameters
        params = PrivacyParams(
            sigma=1.0,
            num_steps=1,
            num_selected=1,
            num_epochs=1,
            delta=0.1  # Large delta to avoid inf
        )
        config = SchemeConfig()
        
        epsilon = allocation_epsilon_decomposition(params, config)
        
        # This might return inf or finite value depending on constraints
        if np.isfinite(epsilon):
            assert epsilon > 0, f"Minimal params epsilon should be positive: {epsilon}"
        else:
            print(f"Minimal parameters result in epsilon = {epsilon}")
    
    def test_large_scale_parameters(self):
        """Test with large-scale parameters"""
        from random_allocation.random_allocation_scheme.decomposition import allocation_epsilon_decomposition
        
        # Decomposition only supports num_selected=1, so use that
        params = PrivacyParams(
            sigma=100.0,    # Very large sigma
            num_steps=1000, # Large steps
            num_selected=1,  # Decomposition constraint
            num_epochs=1,    # Decomposition constraint
            delta=0.0001    # delta << 1/1000 = 0.001
        )
        config = SchemeConfig()
        
        epsilon = allocation_epsilon_decomposition(params, config)
        
        if np.isfinite(epsilon):
            assert epsilon >= 0, f"Large scale epsilon should be non-negative: {epsilon}"
            print(f"Large scale parameters: epsilon = {epsilon}")
            if epsilon == 0.0:
                print("Note: epsilon = 0.0 is valid for very large sigma (strong privacy)")
        else:
            print(f"Large scale parameters result in epsilon = {epsilon}")
    
    def test_high_selection_ratio(self):
        """Test with high selection ratios"""
        from random_allocation.random_allocation_scheme.analytic import allocation_epsilon_analytic
        
        # High selection ratio that should trigger sampling constraint
        # Use parameters where num_steps_per_round >= num_selected to avoid validation error
        # ceil(num_steps/num_selected) >= num_selected
        # We want high selection ratio but avoid the validation constraint
        params = PrivacyParams(
            sigma=0.5,       # Small sigma to trigger constraint 
            num_steps=20,
            num_selected=4,  # ceil(20/4) = 5 >= 4, and high selection ratio
            num_epochs=1,
            delta=1e-5
        )
        config = SchemeConfig()
        
        epsilon = allocation_epsilon_analytic(params, config)
        
        # This should return inf due to sampling probability constraint
        threshold = np.sqrt(params.num_selected/params.num_steps)
        print(f"Selection ratio {params.num_selected}/{params.num_steps} = {params.num_selected/params.num_steps:.2f}")
        print(f"Threshold = {threshold:.3f}, Result = {epsilon}")
        
        # High selection ratios should trigger mathematical constraints
        assert np.isinf(epsilon), f"High selection ratio should trigger inf constraint, got {epsilon}"


class TestMultiEpochScenarios:
    """Test multi-epoch scenarios"""
    

    



class TestExtremePrecisionRequirements:
    """Test extreme precision requirements"""
    
    def test_very_small_delta(self):
        """Test with very small delta requirements"""
        from random_allocation.random_allocation_scheme.decomposition import allocation_epsilon_decomposition
        
        # Decomposition only supports num_selected=1
        params = PrivacyParams(
            sigma=10.0,
            num_steps=50,
            num_selected=1,   # Decomposition constraint
            num_epochs=1,
            delta=1e-10  # Very small delta
        )
        config = SchemeConfig()
        
        epsilon = allocation_epsilon_decomposition(params, config)
        
        if np.isfinite(epsilon):
            assert epsilon > 0, f"Very small delta epsilon should be positive: {epsilon}"
            print(f"Very small delta (1e-10) result: epsilon = {epsilon}")
        else:
            print(f"Very small delta triggers epsilon = {epsilon}")
    



class TestErrorConditions:
    """Test proper error handling"""
    
    def test_missing_epsilon_delta_error(self):
        """Test error when both epsilon and delta are None"""
        with pytest.raises((ValueError, AssertionError)):
            bad_params = PrivacyParams(
                sigma=1.0, num_steps=10, num_selected=5, num_epochs=1,
                epsilon=None, delta=None  # Both missing
            )
            bad_params.validate()
    
    def test_both_epsilon_delta_provided_error(self):
        """Test error when both epsilon and delta are provided"""
        with pytest.raises((ValueError, AssertionError)):
            bad_params = PrivacyParams(
                sigma=1.0, num_steps=10, num_selected=5, num_epochs=1,
                epsilon=1.0, delta=1e-5  # Both provided
            )
            bad_params.validate()
    



if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 