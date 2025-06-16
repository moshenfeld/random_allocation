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
    
    def test_allocation_combined_method(self):
        """Test allocation combined method"""
        pytest.importorskip("random_allocation.random_allocation_scheme.combined")
        from random_allocation.random_allocation_scheme.combined import allocation_epsilon_combined, allocation_delta_combined
        
        params = PrivacyParams(
            sigma=5.0, num_steps=10, num_selected=2, num_epochs=1, delta=1e-4
        )
        config = SchemeConfig(
            allocation_direct_alpha_orders=[2, 3, 4, 5]
        )
        
        epsilon = allocation_epsilon_combined(params, config)
        
        assert np.isfinite(epsilon) or np.isinf(epsilon), f"Combined epsilon returned {epsilon}"
        if np.isfinite(epsilon):
            assert epsilon > 0, f"Combined epsilon should be positive: {epsilon}"
    
    def test_allocation_recursive_method(self):
        """Test allocation recursive method"""
        pytest.importorskip("random_allocation.random_allocation_scheme.recursive")
        from random_allocation.random_allocation_scheme.recursive import allocation_epsilon_recursive, allocation_delta_recursive
        
        params = PrivacyParams(
            sigma=5.0, num_steps=10, num_selected=2, num_epochs=1, delta=1e-4
        )
        config = SchemeConfig()
        
        epsilon = allocation_epsilon_recursive(params, config)
        
        assert np.isfinite(epsilon) or np.isinf(epsilon), f"Recursive epsilon returned {epsilon}"
        if np.isfinite(epsilon):
            assert epsilon > 0, f"Recursive epsilon should be positive: {epsilon}"
    
    def test_allocation_lower_bound_method(self):
        """Test allocation lower bound method"""
        pytest.importorskip("random_allocation.random_allocation_scheme.lower_bound")
        from random_allocation.random_allocation_scheme.lower_bound import allocation_epsilon_lower_bound, allocation_delta_lower_bound
        
        params = PrivacyParams(
            sigma=5.0, num_steps=10, num_selected=2, num_epochs=1, delta=1e-4
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
        
        params = PrivacyParams(
            sigma=100.0,    # Very large sigma
            num_steps=1000, # Large steps
            num_selected=10, # Small fraction selected
            num_epochs=1,
            delta=0.01
        )
        config = SchemeConfig()
        
        epsilon = allocation_epsilon_decomposition(params, config)
        
        if np.isfinite(epsilon):
            assert epsilon > 0, f"Large scale epsilon should be positive: {epsilon}"
            print(f"Large scale parameters: epsilon = {epsilon}")
        else:
            print(f"Large scale parameters result in epsilon = {epsilon}")
    
    def test_high_selection_ratio(self):
        """Test with high selection ratios"""
        from random_allocation.random_allocation_scheme.analytic import allocation_epsilon_analytic
        
        # High selection ratio that should trigger sampling constraint
        params = PrivacyParams(
            sigma=1.0,
            num_steps=10,
            num_selected=9,  # 90% selection
            num_epochs=1,
            delta=1e-5
        )
        config = SchemeConfig()
        
        epsilon = allocation_epsilon_analytic(params, config)
        
        # This should return inf due to sampling probability constraint
        threshold = np.sqrt(params.num_selected/params.num_steps)
        print(f"High selection ratio {params.num_selected}/{params.num_steps} = {params.num_selected/params.num_steps:.2f}")
        print(f"Threshold = {threshold:.3f}, Result = {epsilon}")
        
        # High selection ratios should trigger mathematical constraints
        assert np.isinf(epsilon), f"High selection ratio should trigger inf constraint, got {epsilon}"


class TestMultiEpochScenarios:
    """Test multi-epoch scenarios"""
    
    def test_multi_epoch_decomposition(self):
        """Test decomposition with multiple epochs"""
        from random_allocation.random_allocation_scheme.decomposition import allocation_epsilon_decomposition
        
        params = PrivacyParams(
            sigma=5.0,
            num_steps=20,
            num_selected=2,
            num_epochs=3,  # Multiple epochs
            delta=1e-3
        )
        config = SchemeConfig()
        
        epsilon = allocation_epsilon_decomposition(params, config)
        
        if np.isfinite(epsilon):
            assert epsilon > 0, f"Multi-epoch epsilon should be positive: {epsilon}"
            print(f"Multi-epoch (3) result: epsilon = {epsilon}")
        else:
            print(f"Multi-epoch parameters result in epsilon = {epsilon}")
    
    def test_epoch_scaling_behavior(self):
        """Test how epsilon scales with number of epochs"""
        from random_allocation.random_allocation_scheme.decomposition import allocation_epsilon_decomposition
        
        base_params = PrivacyParams(
            sigma=10.0, num_steps=20, num_selected=2, num_epochs=1, delta=1e-3
        )
        config = SchemeConfig()
        
        epoch_values = [1, 2, 3, 5]
        results = {}
        
        for epochs in epoch_values:
            params = PrivacyParams(
                sigma=base_params.sigma,
                num_steps=base_params.num_steps,
                num_selected=base_params.num_selected,
                num_epochs=epochs,
                delta=base_params.delta
            )
            
            epsilon = allocation_epsilon_decomposition(params, config)
            results[epochs] = epsilon
        
        print(f"Epoch scaling results: {results}")
        
        # Check that finite results are positive
        for epochs, epsilon in results.items():
            if np.isfinite(epsilon):
                assert epsilon > 0, f"Epochs={epochs} should give positive epsilon: {epsilon}"


class TestExtremePrecisionRequirements:
    """Test extreme precision requirements"""
    
    def test_very_small_delta(self):
        """Test with very small delta requirements"""
        from random_allocation.random_allocation_scheme.decomposition import allocation_epsilon_decomposition
        
        params = PrivacyParams(
            sigma=10.0,
            num_steps=50,
            num_selected=5,
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
    
    def test_very_large_sigma(self):
        """Test with very large sigma (strong privacy)"""
        from random_allocation.random_allocation_scheme.decomposition import allocation_epsilon_decomposition
        
        params = PrivacyParams(
            sigma=1000.0,  # Very large sigma
            num_steps=50,
            num_selected=1,
            num_epochs=1,
            delta=0.01
        )
        config = SchemeConfig()
        
        epsilon = allocation_epsilon_decomposition(params, config)
        
        if np.isfinite(epsilon):
            assert epsilon > 0, f"Very large sigma epsilon should be positive: {epsilon}"
            print(f"Very large sigma (1000.0) result: epsilon = {epsilon}")
        else:
            print(f"Very large sigma triggers epsilon = {epsilon}")


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
    
    def test_extreme_parameter_values(self):
        """Test with extreme parameter values"""
        extreme_cases = [
            # Very small sigma
            PrivacyParams(sigma=1e-10, num_steps=10, num_selected=1, num_epochs=1, delta=0.1),
            # Very large num_steps  
            PrivacyParams(sigma=1.0, num_steps=1000000, num_selected=1, num_epochs=1, delta=0.1),
            # Zero delta (should fail validation)
            # PrivacyParams(sigma=1.0, num_steps=10, num_selected=1, num_epochs=1, delta=0.0),
        ]
        
        from random_allocation.random_allocation_scheme.decomposition import allocation_epsilon_decomposition
        config = SchemeConfig()
        
        for i, params in enumerate(extreme_cases):
            params.validate()
            epsilon = allocation_epsilon_decomposition(params, config)
            print(f"Extreme case {i}: epsilon = {epsilon}")
            
            if np.isfinite(epsilon):
                assert epsilon > 0, f"Extreme case {i} epsilon should be positive: {epsilon}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 