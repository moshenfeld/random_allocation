#!/usr/bin/env python3
"""
Basic Functionality Tests - Level 1

These tests check the most fundamental operations without hiding any failures.
Any failure here indicates a critical bug that must be fixed before proceeding.
"""

import pytest
import numpy as np
from random_allocation.comparisons.definitions import PrivacyParams, SchemeConfig, Direction
from random_allocation.other_schemes.local import Gaussian_epsilon, Gaussian_delta


class TestGaussianMechanismBaseline:
    """Test the fundamental Gaussian mechanism as our baseline"""
    
    def test_gaussian_epsilon_basic(self):
        """Test basic Gaussian epsilon calculation"""
        sigma = 1.0
        delta = 1e-5
        
        epsilon = Gaussian_epsilon(sigma, delta)
        
        # No exceptions, no hiding - these must pass
        assert np.isfinite(epsilon), f"Gaussian_epsilon returned {epsilon}, should be finite"
        assert epsilon > 0, f"Gaussian_epsilon returned {epsilon}, should be positive"
    
    def test_gaussian_delta_basic(self):
        """Test basic Gaussian delta calculation"""
        sigma = 1.0
        epsilon = 1.0
        
        delta = Gaussian_delta(sigma, epsilon)
        
        # No exceptions, no hiding - these must pass
        assert np.isfinite(delta), f"Gaussian_delta returned {delta}, should be finite"
        assert 0 < delta < 1, f"Gaussian_delta returned {delta}, should be in (0,1)"
    
    def test_gaussian_round_trip(self):
        """Test epsilon -> delta -> epsilon consistency"""
        sigma = 1.0
        original_delta = 1e-5
        
        epsilon = Gaussian_epsilon(sigma, original_delta)
        computed_delta = Gaussian_delta(sigma, epsilon)
        
        # Check round-trip consistency
        relative_error = abs(computed_delta - original_delta) / original_delta
        assert relative_error < 0.01, f"Round-trip error {relative_error} too large"


class TestParameterValidation:
    """Test parameter validation without hiding failures"""
    
    def test_privacy_params_creation(self):
        """Test creating valid PrivacyParams"""
        # This should work without issues
        params = PrivacyParams(
            sigma=1.0,
            num_steps=10,
            num_selected=5,
            num_epochs=1,
            delta=1e-5
        )
        
        # Basic validation
        assert params.sigma == 1.0
        assert params.num_steps == 10
        assert params.num_selected == 5
        assert params.num_epochs == 1
        assert params.delta == 1e-5
    
    def test_privacy_params_validation(self):
        """Test parameter validation catches invalid inputs during object creation"""
        # This should work (validation happens in __post_init__)
        valid_params = PrivacyParams(
            sigma=1.0, num_steps=10, num_selected=5, num_epochs=1, delta=1e-5
        )
        # No need to call validate() - it happens automatically
        
        # These should fail during object creation
        with pytest.raises(ValueError):
            PrivacyParams(
                sigma=-1.0,  # Invalid: negative sigma
                num_steps=10, num_selected=5, num_epochs=1, delta=1e-5
            )
        
        with pytest.raises(ValueError):
            PrivacyParams(
                sigma=1.0, num_steps=0,  # Invalid: zero steps
                num_selected=5, num_epochs=1, delta=1e-5
            )
        
        with pytest.raises(ValueError):
            PrivacyParams(
                sigma=1.0, num_steps=10, num_selected=15,  # Invalid: selected > steps
                num_epochs=1, delta=1e-5
            )
    
    def test_scheme_config_creation(self):
        """Test creating SchemeConfig"""
        config = SchemeConfig()
        
        # Should have reasonable defaults
        assert config.discretization > 0
        assert config.delta_tolerance > 0
        assert config.epsilon_tolerance > 0


class TestDirectionEnum:
    """Test Direction enum functionality"""
    
    def test_direction_values(self):
        """Test Direction enum has correct values"""
        assert Direction.ADD.value == 'add'
        assert Direction.REMOVE.value == 'remove'
        assert Direction.BOTH.value == 'both'
    
    def test_direction_usage(self):
        """Test Direction enum can be used in basic operations"""
        directions = [Direction.ADD, Direction.REMOVE, Direction.BOTH]
        
        for direction in directions:
            # Should be able to use direction without issues
            assert isinstance(direction, Direction)
            assert isinstance(direction.value, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 