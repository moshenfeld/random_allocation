#!/usr/bin/env python3
"""
Basic Functionality Tests - Level 1

These tests check the most fundamental operations without hiding any failures.
Any failure here indicates a critical bug that must be fixed before proceeding.
"""

import pytest
import numpy as np
import os
from random_allocation.comparisons.definitions import PrivacyParams, SchemeConfig, Direction
from random_allocation.other_schemes.local import Gaussian_epsilon, Gaussian_delta
from tests.test_utils import ResultsReporter


@pytest.fixture(scope="session")
def reporter() -> ResultsReporter:
    """Set up the results reporter for the session."""
    return ResultsReporter("test_basic_01_functionality")


@pytest.fixture(scope="session", autouse=True)
def session_teardown(reporter: ResultsReporter):
    """Teardown fixture to save results at the end of the session."""
    yield
    
    # Save results - but only if not running as part of suite
    is_suite_run = os.environ.get('PYTEST_SUITE_RUN', 'false').lower() == 'true'
    
    if is_suite_run:
        # Just finalize results for suite collection
        reporter.get_results()
    else:
        # Save individual JSON file when run standalone
        reporter.finalize_and_save()


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
        
        # Basic validation 
        assert np.isfinite(delta), f"Gaussian_delta returned {delta}, should be finite"
        assert 0 < delta < 1, f"Gaussian_delta returned {delta}, should be in (0,1)"
    
    def test_gaussian_round_trip(self):
        """Test round-trip consistency: epsilon -> delta -> epsilon"""
        sigma = 1.0
        original_epsilon = 1.0
        
        # Convert epsilon to delta
        delta = Gaussian_delta(sigma, original_epsilon)
        
        # Convert delta back to epsilon  
        recovered_epsilon = Gaussian_epsilon(sigma, delta)
        
        # Check round-trip consistency
        relative_error = abs(original_epsilon - recovered_epsilon) / original_epsilon
        assert relative_error < 0.01, f"Round-trip error too large: {relative_error}"


class TestPrivacyParameterCreation:
    """Test privacy parameter validation and creation"""
    
    def test_privacy_params_creation(self):
        """Test basic PrivacyParams creation"""
        # Test with epsilon
        params_eps = PrivacyParams(sigma=1.0, num_steps=10, epsilon=1.0)
        assert params_eps.epsilon == 1.0
        assert params_eps.delta is None
        
        # Test with delta
        params_delta = PrivacyParams(sigma=1.0, num_steps=10, delta=1e-6)
        assert params_delta.delta == 1e-6
        assert params_delta.epsilon is None
        
        # Test defaults
        assert params_eps.num_epochs == 1
    
    def test_privacy_params_validation(self):
        """Test PrivacyParams validation rules"""
        # Test invalid sigma
        with pytest.raises((ValueError, AssertionError)):
            PrivacyParams(sigma=-1.0, num_steps=10, epsilon=1.0)
        
        with pytest.raises((ValueError, AssertionError)):
            PrivacyParams(sigma=0.0, num_steps=10, epsilon=1.0)
        
        # Test invalid num_steps
        with pytest.raises((ValueError, AssertionError)):
            PrivacyParams(sigma=1.0, num_steps=0, epsilon=1.0)
        
        with pytest.raises((ValueError, AssertionError)):
            PrivacyParams(sigma=1.0, num_steps=-5, epsilon=1.0)
        
        # Test invalid epsilon
        with pytest.raises((ValueError, AssertionError)):
            PrivacyParams(sigma=1.0, num_steps=10, epsilon=-1.0)
        
        # Test invalid delta  
        with pytest.raises((ValueError, AssertionError)):
            PrivacyParams(sigma=1.0, num_steps=10, delta=-1e-6)
        
        with pytest.raises((ValueError, AssertionError)):
            PrivacyParams(sigma=1.0, num_steps=10, delta=1.5)  # delta > 1


class TestSchemeConfigCreation:
    """Test scheme configuration validation"""
    
    def test_scheme_config_creation(self):
        """Test basic SchemeConfig creation"""
        config = SchemeConfig()
        
        # Check defaults exist
        assert hasattr(config, 'discretization')
        assert hasattr(config, 'epsilon_tolerance') 
        assert hasattr(config, 'delta_tolerance')
        
        # Test custom values
        config_custom = SchemeConfig(discretization=0.001, epsilon_tolerance=0.01)
        assert config_custom.discretization == 0.001
        assert config_custom.epsilon_tolerance == 0.01


class TestDirectionEnum:
    """Test Direction enum functionality"""
    
    def test_direction_values(self):
        """Test Direction enum has correct values"""
        assert Direction.ADD.value == "add"
        assert Direction.REMOVE.value == "remove" 
        assert Direction.BOTH.value == "both"
    
    def test_direction_usage(self):
        """Test Direction enum can be used in function calls"""
        # This is mainly testing the enum is properly importable and usable
        directions = [Direction.ADD, Direction.REMOVE, Direction.BOTH]
        
        for direction in directions:
            assert isinstance(direction.value, str)
            assert direction.value in ["add", "remove", "both"]


class TestBasicAllocationFunctionality:
    """Test basic allocation scheme functionality"""
    
    def test_privacy_params_with_custom_values(self):
        """Test PrivacyParams with various custom values"""
        # Test with epsilon
        params_eps = PrivacyParams(
            sigma=2.0,
            num_steps=20,
            num_selected=10,
            num_epochs=2,
            epsilon=0.5
        )
        
        assert params_eps.sigma == 2.0
        assert params_eps.num_steps == 20
        assert params_eps.num_selected == 10
        assert params_eps.num_epochs == 2
        assert params_eps.epsilon == 0.5
        assert params_eps.delta is None
        
        # Test with delta
        params_delta = PrivacyParams(
            sigma=1.5,
            num_steps=15,
            num_selected=8,
            num_epochs=1,
            delta=1e-6
        )
        
        assert params_delta.sigma == 1.5
        assert params_delta.num_steps == 15
        assert params_delta.num_selected == 8
        assert params_delta.num_epochs == 1
        assert params_delta.delta == 1e-6
        assert params_delta.epsilon is None
    
    def test_scheme_config_custom_values(self):
        """Test SchemeConfig with custom values"""
        config = SchemeConfig(
            discretization=0.001,
            epsilon_tolerance=0.005,
            delta_tolerance=1e-7
        )
        
        assert config.discretization == 0.001
        assert config.epsilon_tolerance == 0.005
        assert config.delta_tolerance == 1e-7


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 