#!/usr/bin/env python3
"""
Type Annotations Tests

Tests type annotations and type checking functionality for the Random Allocation project.
Based on the type annotations guide.
"""

import pytest
import numpy as np
from typing import Dict, List, Optional, Union, Callable, Any, TypeVar, cast

from random_allocation.comparisons.definitions import PrivacyParams, SchemeConfig, Direction, Verbosity
from random_allocation.random_allocation_scheme.decomposition import allocation_epsilon_decomposition, allocation_delta_decomposition
from random_allocation.other_schemes.local import Gaussian_epsilon, Gaussian_delta


class TestTypeAnnotationCompliance:
    """Test that functions respect their type annotations"""
    
    def test_privacy_params_type_annotations(self):
        """Test PrivacyParams type annotations"""
        # Valid types
        params = PrivacyParams(
            sigma=1.0,          # float
            num_steps=10,       # int
            num_selected=5,     # int
            num_epochs=1,       # int
            delta=1e-5          # Optional[float]
        )
        
        # Check types after construction
        assert isinstance(params.sigma, float), f"sigma should be float, got {type(params.sigma)}"
        assert isinstance(params.num_steps, int), f"num_steps should be int, got {type(params.num_steps)}"
        assert isinstance(params.num_selected, int), f"num_selected should be int, got {type(params.num_selected)}"
        assert isinstance(params.num_epochs, int), f"num_epochs should be int, got {type(params.num_epochs)}"
        assert isinstance(params.delta, float), f"delta should be float, got {type(params.delta)}"
        assert params.epsilon is None, f"epsilon should be None when not provided"
    
    def test_scheme_config_type_annotations(self):
        """Test SchemeConfig type annotations"""
        config = SchemeConfig(
            discretization=1e-4,                    # float
            allocation_direct_alpha_orders=[2, 3], # Optional[List[int]]
            delta_tolerance=1e-15,                  # float
            verbosity=Verbosity.WARNINGS           # Verbosity enum
        )
        
        assert isinstance(config.discretization, float)
        assert isinstance(config.allocation_direct_alpha_orders, list)
        assert all(isinstance(x, int) for x in config.allocation_direct_alpha_orders)
        assert isinstance(config.delta_tolerance, float)
        assert isinstance(config.verbosity, Verbosity)
    
    def test_direction_enum_annotations(self):
        """Test Direction enum type annotations"""
        directions = [Direction.ADD, Direction.REMOVE, Direction.BOTH]
        
        for direction in directions:
            assert isinstance(direction, Direction)
            assert isinstance(direction.value, str)
    
    def test_function_return_types(self):
        """Test that functions return expected types"""
        params = PrivacyParams(sigma=2.0, num_steps=10, num_selected=1, num_epochs=1, delta=1e-5)  # decomposition requires num_selected=1
        config = SchemeConfig()
        
        # Test Gaussian functions return floats
        epsilon = Gaussian_epsilon(sigma=1.0, delta=1e-6)
        delta = Gaussian_delta(sigma=1.0, epsilon=1.0)
        
        assert isinstance(epsilon, (int, float)), f"Gaussian_epsilon should return numeric, got {type(epsilon)}"
        assert isinstance(delta, (int, float)), f"Gaussian_delta should return numeric, got {type(delta)}"
        
        # Test allocation functions
        alloc_epsilon = allocation_epsilon_decomposition(params, config)
        assert isinstance(alloc_epsilon, (int, float)), f"allocation_epsilon_decomposition should return numeric, got {type(alloc_epsilon)}"


class TestTypeConversions:
    """Test type conversions in PrivacyParams"""
    
    def test_automatic_type_conversion(self):
        """Test that PrivacyParams converts types automatically"""
        # Input various numeric types
        params = PrivacyParams(
            sigma="2.5",        # String -> float
            num_steps=10.0,     # Float -> int  
            num_selected=np.int32(5),  # numpy int -> int
            num_epochs=np.float64(1.0), # numpy float -> int
            delta="1e-5"        # String -> float
        )
        
        # Check all conversions worked
        assert isinstance(params.sigma, float) and params.sigma == 2.5
        assert isinstance(params.num_steps, int) and params.num_steps == 10
        assert isinstance(params.num_selected, int) and params.num_selected == 5
        assert isinstance(params.num_epochs, int) and params.num_epochs == 1
        assert isinstance(params.delta, float) and params.delta == 1e-5
    
    def test_invalid_type_conversion(self):
        """Test that invalid types are caught"""
        # These should fail during validation
        with pytest.raises((ValueError, TypeError)):
            params = PrivacyParams(
                sigma="invalid",  # Can't convert to float
                num_steps=10, num_selected=5, num_epochs=1, delta=1e-5
            )
            params.validate()


class TestCallableTypeAnnotations:
    """Test callable type annotations"""
    
    def test_epsilon_calculator_type(self):
        """Test EpsilonCalculator type annotation - REMOVED"""
        # This functionality was intentionally removed - test passes as no-op
        # If this functionality is needed again, implement proper type alias testing
        pass
    
    def test_delta_calculator_type(self):
        """Test DeltaCalculator type annotation - REMOVED"""
        # This functionality was intentionally removed - test passes as no-op
        # If this functionality is needed again, implement proper type alias testing
        pass


class TestOptionalAndUnionTypes:
    """Test Optional and Union type handling"""
    
    def test_optional_parameters(self):
        """Test Optional parameter handling"""
        # Only epsilon provided
        params1 = PrivacyParams(
            sigma=1.0, num_steps=10, num_selected=5, num_epochs=1,
            epsilon=1.0, delta=None
        )
        assert params1.epsilon == 1.0
        assert params1.delta is None
        
        # Only delta provided  
        params2 = PrivacyParams(
            sigma=1.0, num_steps=10, num_selected=5, num_epochs=1,
            epsilon=None, delta=1e-5
        )
        assert params2.epsilon is None
        assert params2.delta == 1e-5
    
    def test_union_type_handling(self):
        """Test Union type handling in function returns"""
        # Functions might return float or inf
        # Need num_selected <= ceil(num_steps/num_selected) for analytic method
        params = PrivacyParams(sigma=0.1, num_steps=100, num_selected=8, num_epochs=1, delta=1e-6)
        config = SchemeConfig()
        
        from random_allocation.random_allocation_scheme.analytic import allocation_epsilon_analytic
        result = allocation_epsilon_analytic(params, config)
        
        # Should be Union[float, inf] - both are valid
        assert isinstance(result, (int, float)), f"Should return numeric type, got {type(result)}"
        # inf is still a float in Python
        assert np.isfinite(result) or np.isinf(result), f"Should be finite or inf, got {result}"


class TestGenericTypes:
    """Test generic type usage"""
    
    def test_type_variable_usage(self):
        """Test TypeVar usage in type annotations"""
        # Test that we can use generic types properly
        T = TypeVar('T', bound=float)
        
        def process_numeric(value: T) -> T:
            return value
        
        # Should work with different numeric types
        result_float = process_numeric(1.5)
        result_int = process_numeric(1)
        
        assert isinstance(result_float, float)
        assert isinstance(result_int, int)
    
    def test_list_type_annotations(self):
        """Test List type annotations"""
        alpha_orders: List[int] = [2, 3, 4, 5]
        
        # Should be a list of ints
        assert isinstance(alpha_orders, list)
        assert all(isinstance(x, int) for x in alpha_orders)
        
        # Use in SchemeConfig
        config = SchemeConfig(allocation_direct_alpha_orders=alpha_orders)
        assert config.allocation_direct_alpha_orders == alpha_orders
    
    def test_dict_type_annotations(self):
        """Test Dict type annotations"""
        results: Dict[str, float] = {
            "epsilon": 1.5,
            "delta": 1e-5
        }
        
        assert isinstance(results, dict)
        assert all(isinstance(k, str) for k in results.keys())
        assert all(isinstance(v, (int, float)) for v in results.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 