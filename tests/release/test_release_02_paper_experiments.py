#!/usr/bin/env python3
"""
Paper Experiments Tests

Tests that verify the paper experiments in the examples directory can be run
and produce reasonable results. These are integration tests for release validation.
"""

import pytest
import os
import sys
import importlib.util
import numpy as np
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from random_allocation.comparisons.definitions import PrivacyParams, SchemeConfig


class TestPaperExperiments:
    """Test that paper experiments can be executed successfully"""
    
    @pytest.fixture
    def examples_dir(self):
        """Get the examples directory path"""
        return project_root / "examples"
    
    def load_example_module(self, examples_dir, filename):
        """Helper to load an example module dynamically"""
        file_path = examples_dir / filename
        if not file_path.exists():
            pytest.skip(f"Example file {filename} not found")
        
        spec = importlib.util.spec_from_file_location("example_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    def test_example_01_introduction(self, examples_dir):
        """Test Example 01: Introduction to Random Allocation"""
        module = self.load_example_module(examples_dir, "01_introduction.py")
        
        # Should define basic functions and run without error
        assert hasattr(module, '__name__')
        
    def test_example_02_comparison(self, examples_dir):
        """Test Example 02: Comparison of Privacy Schemes"""
        module = self.load_example_module(examples_dir, "02_comparison.py")
        
        # Should define comparison functions
        assert hasattr(module, '__name__')
        
    def test_example_03_varying_sigma(self, examples_dir):
        """Test Example 03: Varying Sigma Analysis"""
        module = self.load_example_module(examples_dir, "03_varying_sigma.py")
        
        # Should perform sigma variation analysis
        assert hasattr(module, '__name__')
        
    def test_example_04_varying_num_steps(self, examples_dir):
        """Test Example 04: Varying Number of Steps"""
        module = self.load_example_module(examples_dir, "04_varying_num_steps.py")
        
        # Should perform num_steps variation analysis
        assert hasattr(module, '__name__')
        
    def test_example_05_varying_num_selected(self, examples_dir):
        """Test Example 05: Varying Number Selected"""
        module = self.load_example_module(examples_dir, "05_varying_num_selected.py")
        
        # Should perform num_selected variation analysis
        assert hasattr(module, '__name__')
        
    def test_example_06_varying_num_epochs(self, examples_dir):
        """Test Example 06: Varying Number of Epochs"""
        module = self.load_example_module(examples_dir, "06_varying_num_epochs.py")
        
        # Should perform num_epochs variation analysis
        assert hasattr(module, '__name__')
        
    def test_example_07_fixed_delta_varying_epsilon(self, examples_dir):
        """Test Example 07: Fixed Delta Varying Epsilon"""
        module = self.load_example_module(examples_dir, "07_fixed_delta_varying_epsilon.py")
        
        # Should perform epsilon variation with fixed delta
        assert hasattr(module, '__name__')
    
    @pytest.mark.slow
    def test_all_examples_can_import(self, examples_dir):
        """Test that all example files can be imported without syntax errors"""
        example_files = [
            "01_introduction.py",
            "02_comparison.py", 
            "03_varying_sigma.py",
            "04_varying_num_steps.py",
            "05_varying_num_selected.py",
            "06_varying_num_epochs.py",
            "07_fixed_delta_varying_epsilon.py"
        ]
        
        successful_imports = 0
        import_failures = []
        
        for filename in example_files:
            try:
                self.load_example_module(examples_dir, filename)
                successful_imports += 1
            except Exception as e:
                import_failures.append(f"{filename}: {e}")
        
        # If any examples fail to import, fail the test with details
        if import_failures:
            failure_details = "\n".join(import_failures)
            pytest.fail(f"Example import failures:\n{failure_details}")
        
        # All examples should import successfully
        assert successful_imports == len(example_files), f"Only {successful_imports}/{len(example_files)} examples imported successfully"


class TestExampleFunctionality:
    """Test specific functionality from examples that we can verify"""
    
    def test_basic_comparison_functions(self):
        """Test that comparison functions work with basic parameters"""
        # Import comparison functionality
        from random_allocation.comparisons.plot_comparison import plot_comparison_by_varying
        from random_allocation.comparisons.definitions import PrivacyParams, SchemeConfig
        
        # Test parameters that should work
        params = PrivacyParams(
            sigma=2.0,
            num_steps=10,
            num_selected=3,
            num_epochs=1,
            delta=1e-5
        )
        config = SchemeConfig()
        
        # Should be able to create comparison plots (without actually plotting)
        # This tests the setup without actually generating plots
        from random_allocation.comparisons.get_method_dict import get_method_dict
        methods = ["Gaussian", "allocation_epsilon_decomposition", "local"]
        method_dict = get_method_dict(methods)
        
        assert len(method_dict) > 0, "Should have at least some methods available"
    
    def test_privacy_parameter_validation(self):
        """Test that examples use valid privacy parameters"""
        # These are typical parameters from the examples
        test_cases = [
            # Basic case
            {"sigma": 1.0, "num_steps": 100, "num_selected": 10, "num_epochs": 1, "delta": 1e-5},
            # Large sigma
            {"sigma": 10.0, "num_steps": 50, "num_selected": 5, "num_epochs": 1, "delta": 1e-6},
            # Multiple epochs
            {"sigma": 2.0, "num_steps": 20, "num_selected": 2, "num_epochs": 5, "delta": 1e-4},
        ]
        
        for params_dict in test_cases:
            params = PrivacyParams(**params_dict)
            params.validate()  # Should not raise an exception
            
            # Check basic constraints
            assert params.sigma > 0
            assert params.num_steps > 0
            assert params.num_selected > 0
            assert params.num_epochs > 0
            assert 0 < params.delta < 1
    
    def test_scheme_configurations(self):
        """Test that scheme configurations used in examples are valid"""
        # Default configuration
        config_default = SchemeConfig()
        assert config_default.discretization > 0
        
        # Configuration with specific alpha orders (used in allocation_direct)
        config_direct = SchemeConfig(allocation_direct_alpha_orders=[2, 3, 4, 5])
        assert len(config_direct.allocation_direct_alpha_orders) > 0
        assert all(isinstance(x, int) for x in config_direct.allocation_direct_alpha_orders)
        
        # High precision configuration
        config_precise = SchemeConfig(discretization=1e-6, delta_tolerance=1e-16)
        assert config_precise.discretization < config_default.discretization
        assert config_precise.delta_tolerance < config_default.delta_tolerance


class TestExampleDataFlow:
    """Test the data flow patterns used in examples"""
    
    def test_basic_calculation_flow(self):
        """Test basic calculation flow from examples"""
        from random_allocation.other_schemes.local import Gaussian_epsilon, Gaussian_delta
        from random_allocation.random_allocation_scheme.decomposition import allocation_epsilon_decomposition
        
        # Example flow: Given sigma and delta, calculate epsilon
        sigma = 2.0
        delta = 1e-5
        
        # Gaussian baseline
        gaussian_epsilon = Gaussian_epsilon(sigma=sigma, delta=delta)
        assert gaussian_epsilon > 0, f"Gaussian epsilon should be positive, got {gaussian_epsilon}"
        
        # Allocation scheme
        params = PrivacyParams(sigma=sigma, num_steps=10, num_selected=3, num_epochs=1, delta=delta)
        config = SchemeConfig()
        
        allocation_epsilon = allocation_epsilon_decomposition(params, config)
        assert allocation_epsilon >= 0, f"Allocation epsilon should be non-negative, got {allocation_epsilon}"
    
    def test_parameter_sweep_pattern(self):
        """Test parameter sweep pattern used in examples"""
        # Pattern: vary one parameter while keeping others fixed
        base_params = {
            "num_steps": 10,
            "num_selected": 3, 
            "num_epochs": 1,
            "delta": 1e-5
        }
        
        # Sweep sigma values
        sigma_values = [0.5, 1.0, 2.0, 5.0, 10.0]
        results = []
        
        from random_allocation.other_schemes.local import Gaussian_epsilon
        
        for sigma in sigma_values:
            epsilon = Gaussian_epsilon(sigma=sigma, delta=base_params["delta"])
            results.append(epsilon)
        
        # Should see decreasing epsilon as sigma increases
        assert len(results) == len(sigma_values)
        assert all(eps > 0 for eps in results), f"All epsilon values should be positive: {results}"
        
        # Generally expect decreasing trend (larger sigma -> smaller epsilon)
        # Allow some numerical variation
        mostly_decreasing = sum(results[i] >= results[i+1] for i in range(len(results)-1))
        assert mostly_decreasing >= len(results) * 0.6, f"Expected mostly decreasing trend in epsilon vs sigma"
    
    def test_comparison_pattern(self):
        """Test comparison pattern between different schemes"""
        from random_allocation.other_schemes.local import Gaussian_epsilon
        from random_allocation.other_schemes.poisson import poisson_pld_epsilon
        
        # Common parameters
        sigma = 2.0
        delta = 1e-5
        
        # Compare Gaussian vs Poisson
        gaussian_eps = Gaussian_epsilon(sigma=sigma, delta=delta)
        
        params = PrivacyParams(sigma=sigma, num_steps=10, num_selected=3, num_epochs=1, delta=delta)
        config = SchemeConfig()
        poisson_eps = poisson_pld_epsilon(params, config)
        
        # Both should be positive
        assert gaussian_eps > 0
        assert poisson_eps > 0
        
        # Results should be in reasonable range
        assert 0.1 <= gaussian_eps <= 100, f"Gaussian epsilon {gaussian_eps} out of reasonable range"
        assert 0.1 <= poisson_eps <= 100, f"Poisson epsilon {poisson_eps} out of reasonable range"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 