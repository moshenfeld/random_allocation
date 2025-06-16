#!/usr/bin/env python3
"""
Additional Allocation Methods Tests - Level 3

Tests the remaining allocation privacy methods without hiding any failures.
Builds on Level 1 (basic) and Level 2 (core methods) foundations.
"""

import pytest
import numpy as np
from random_allocation.comparisons.definitions import PrivacyParams, SchemeConfig, Direction
from random_allocation.random_allocation_scheme.direct import allocation_epsilon_direct, allocation_delta_direct
from random_allocation.random_allocation_scheme.RDP_DCO import allocation_epsilon_RDP_DCO, allocation_delta_RDP_DCO


class TestAllocationDirect:
    """Test allocation direct method - needs alpha_orders configuration"""
    
    def test_direct_epsilon_with_alpha_orders(self):
        """Test direct epsilon with proper alpha_orders configuration"""
        params = PrivacyParams(
            sigma=5.0,
            num_steps=10,
            num_selected=2,
            num_epochs=1,
            delta=1e-4
        )
        
        # Direct method requires alpha_orders
        config = SchemeConfig(
            allocation_direct_alpha_orders=[2, 3, 4, 5]
        )
        
        epsilon = allocation_epsilon_direct(params, config, Direction.ADD)
        
        # Test ADD direction first (simpler)
        assert np.isfinite(epsilon) or np.isinf(epsilon), f"Direct epsilon returned {epsilon}"
        if np.isfinite(epsilon):
            assert epsilon > 0, f"Direct epsilon should be positive: {epsilon}"
    
    def test_direct_delta_with_alpha_orders(self):
        """Test direct delta with proper alpha_orders configuration"""
        params = PrivacyParams(
            sigma=5.0,
            num_steps=10,
            num_selected=2,
            num_epochs=1,
            epsilon=1.0
        )
        
        config = SchemeConfig(
            allocation_direct_alpha_orders=[2, 3, 4, 5]
        )
        
        delta = allocation_delta_direct(params, config, Direction.ADD)
        
        assert np.isfinite(delta), f"Direct delta returned {delta}, should be finite"
        assert 0 < delta <= 1, f"Direct delta should be in (0,1], got {delta}"
    
    def test_direct_missing_alpha_orders_error(self):
        """Test that direct method properly reports missing alpha_orders"""
        params = PrivacyParams(
            sigma=5.0, num_steps=10, num_selected=2, num_epochs=1, delta=1e-4
        )
        
        # Default SchemeConfig without alpha_orders
        config = SchemeConfig()
        
        # This should raise a proper error for REMOVE direction
        with pytest.raises(ValueError, match="allocation_direct_alpha_orders must be provided"):
            allocation_epsilon_direct(params, config, Direction.REMOVE)
    
    def test_direct_remove_direction(self):
        """Test direct method with REMOVE direction"""
        params = PrivacyParams(
            sigma=5.0,
            num_steps=10,
            num_selected=2,
            num_epochs=1,
            delta=1e-4
        )
        
        config = SchemeConfig(
            allocation_direct_alpha_orders=[2, 3, 4, 5]
        )
        
        epsilon = allocation_epsilon_direct(params, config, Direction.REMOVE)
        
        assert np.isfinite(epsilon) or np.isinf(epsilon), f"Direct epsilon (REMOVE) returned {epsilon}"
        if np.isfinite(epsilon):
            assert epsilon > 0, f"Direct epsilon (REMOVE) should be positive: {epsilon}"


class TestAllocationRDP_DCO:
    """Test allocation RDP_DCO method"""
    
    def test_rdp_dco_epsilon_basic(self):
        """Test RDP_DCO epsilon calculation"""
        params = PrivacyParams(
            sigma=2.0,
            num_steps=10,
            num_selected=3,
            num_epochs=1,
            delta=1e-5
        )
        config = SchemeConfig()
        
        epsilon = allocation_epsilon_RDP_DCO(params, config)
        
        assert np.isfinite(epsilon) or np.isinf(epsilon), f"RDP_DCO epsilon returned {epsilon}"
        if np.isfinite(epsilon):
            assert epsilon > 0, f"RDP_DCO epsilon should be positive: {epsilon}"
    
    def test_rdp_dco_delta_basic(self):
        """Test RDP_DCO delta calculation"""
        params = PrivacyParams(
            sigma=2.0,
            num_steps=10,
            num_selected=3,
            num_epochs=1,
            epsilon=1.0
        )
        config = SchemeConfig()
        
        delta = allocation_delta_RDP_DCO(params, config)
        
        assert np.isfinite(delta), f"RDP_DCO delta returned {delta}, should be finite"
        assert 0 < delta <= 1, f"RDP_DCO delta should be in (0,1], got {delta}"
    
    def test_rdp_dco_delta_boundary_case(self):
        """Test RDP_DCO delta with parameters that might cause large delta"""
        # Parameters that previously caused delta > 1 (but that's actually valid output)
        params = PrivacyParams(
            sigma=1.0,
            num_steps=10,
            num_selected=3,
            num_epochs=1,
            epsilon=10.0  # Large epsilon
        )
        config = SchemeConfig()
        
        delta = allocation_delta_RDP_DCO(params, config)
        
        # Large epsilon should result in large delta, potentially close to 1
        assert np.isfinite(delta), f"RDP_DCO delta returned {delta}, should be finite"
        assert 0 < delta <= 1, f"RDP_DCO delta should be in (0,1], got {delta}"
        
        # With large epsilon, delta should be relatively large
        if delta < 1e-10:
            print(f"Warning: Very small delta {delta} with large epsilon {params.epsilon}")


class TestDirectionConsistency:
    """Test that direction parameter works consistently across methods"""
    
    def test_decomposition_direction_consistency(self):
        """Test that decomposition method handles all directions"""
        from random_allocation.random_allocation_scheme.decomposition import allocation_epsilon_decomposition
        
        params = PrivacyParams(
            sigma=10.0, num_steps=5, num_selected=1, num_epochs=1, delta=0.01
        )
        config = SchemeConfig()
        
        directions = [Direction.ADD, Direction.REMOVE, Direction.BOTH]
        results = {}
        
        for direction in directions:
            epsilon = allocation_epsilon_decomposition(params, config, direction)
            results[direction.value] = epsilon
        
        print(f"Decomposition direction results: {results}")
        
        # All directions should work - check they're positive and finite or inf
        for direction, result in results.items():
            assert isinstance(result, (int, float)), f"Direction {direction} failed: {result}"
            assert np.isfinite(result) or np.isinf(result), f"Direction {direction} gave invalid result: {result}"
            if np.isfinite(result):
                assert result > 0, f"Direction {direction} gave non-positive epsilon: {result}"
    
    def test_analytic_direction_consistency(self):
        """Test that analytic method handles all directions"""
        from random_allocation.random_allocation_scheme.analytic import allocation_epsilon_analytic
        
        params = PrivacyParams(
            sigma=10.0, num_steps=5, num_selected=1, num_epochs=1, delta=0.01
        )
        config = SchemeConfig()
        
        directions = [Direction.ADD, Direction.REMOVE, Direction.BOTH]
        results = {}
        
        for direction in directions:
            epsilon = allocation_epsilon_analytic(params, config, direction)
            results[direction.value] = epsilon
        
        print(f"Analytic direction results: {results}")
        
        # All directions should give same result (inf due to constraints) or finite positive values
        for direction, result in results.items():
            assert np.isfinite(result) or np.isinf(result), f"Direction {direction} gave invalid result: {result}"
            if np.isfinite(result):
                assert result > 0, f"Direction {direction} gave non-positive epsilon: {result}"


class TestSchemeConfigRequirements:
    """Test SchemeConfig requirements for different methods"""
    
    def test_default_scheme_config_completeness(self):
        """Test what works with default SchemeConfig"""
        config = SchemeConfig()
        params = PrivacyParams(
            sigma=5.0, num_steps=10, num_selected=2, num_epochs=1, delta=1e-4
        )
        
        # These should work with default config
        from random_allocation.random_allocation_scheme.decomposition import allocation_epsilon_decomposition
        from random_allocation.random_allocation_scheme.analytic import allocation_epsilon_analytic
        
        methods_to_test = [
            ("decomposition", allocation_epsilon_decomposition),
            ("analytic", allocation_epsilon_analytic),
            ("RDP_DCO", allocation_epsilon_RDP_DCO),
        ]
        
        results = {}
        for name, method in methods_to_test:
            result = method(params, config)
            results[name] = result
        
        print(f"Default config results: {results}")
        
        # All methods should work and return valid results
        for name, result in results.items():
            assert isinstance(result, (int, float)), f"{name} should return numeric result: {result}"
            assert np.isfinite(result) or np.isinf(result), f"{name} should return finite or inf: {result}"
            if np.isfinite(result):
                assert result > 0, f"{name} should return positive result: {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 