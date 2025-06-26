#!/usr/bin/env python3
"""
Utility Functions Tests - Level 2

Tests for core utility functions used throughout the random allocation package.
These tests were migrated from search_code_tests.ipynb to ensure utility functions
are properly tested in the main test suite.
"""

import pytest
import numpy as np
from random_allocation.comparisons.utils import search_function_with_bounds, FunctionType, BoundType


class TestSearchFunctionBounds:
    """Test the search_function_with_bounds utility function"""
    
    def test_solution_out_of_bounds(self):
        """Test cases where the target is outside the function's range within bounds"""

        # Case 1: Increasing function with target below minimum
        def increasing_func(x):
            return 2 * x + 5

        # Bounds: (0, 10), which means y ranges from (5, 25)
        # Target y = 3 is below the minimum, so should return None when no bound type helps
        result = search_function_with_bounds(
            func=increasing_func,
            y_target=3,
            bounds=(0, 10),
            tolerance=1e-6,
            function_type=FunctionType.INCREASING,
            bound_type=BoundType.NONE  # No bound constraint
        )

        assert result is None, "Should return None when target is below minimum and no bound constraint"

        # Case 2: Increasing function with target above maximum
        # With BoundType.UPPER (default), it should return the boundary value that best satisfies the constraint
        result = search_function_with_bounds(
            func=increasing_func,
            y_target=30,
            bounds=(0, 10),
            tolerance=1e-6,
            function_type=FunctionType.INCREASING,
            bound_type=BoundType.UPPER  # Explicit upper bound constraint
        )

        # With UPPER bound, it returns x_max=10 because f(10)=25 ≤ 30 (satisfies upper bound)
        assert result == 10, f"With UPPER bound constraint, should return x_max=10, got {result}"
        
        # Case 3: Same case but with no bound constraint should return None
        result = search_function_with_bounds(
            func=increasing_func,
            y_target=30,
            bounds=(0, 10),
            tolerance=1e-6,
            function_type=FunctionType.INCREASING,
            bound_type=BoundType.NONE  # No bound constraint
        )

        assert result is None, "Should return None when target is above maximum and no bound constraint"
         # Case 3: Decreasing function with target below minimum
        def decreasing_func(x):
            return 20 - 2 * x

        # Bounds: (0, 10), which means y ranges from (20, 0)
        # Target y = -5 is below the minimum, so should return None when no bound constraint
        result = search_function_with_bounds(
            func=decreasing_func,
            y_target=-5,
            bounds=(0, 10),
            tolerance=1e-6,
            function_type=FunctionType.DECREASING,
            bound_type=BoundType.NONE  # No bound constraint
        )

        assert result is None, "Should return None when target is below minimum for decreasing function and no bound constraint"

        # Case 4: Decreasing function with target above maximum
        # With UPPER bound constraint, it returns the boundary value that best satisfies f(x) ≤ y_target
        result = search_function_with_bounds(
            func=decreasing_func,
            y_target=25,
            bounds=(0, 10),
            tolerance=1e-6,
            function_type=FunctionType.DECREASING,
            bound_type=BoundType.UPPER  # Explicit upper bound constraint
        )

        # With UPPER bound, it returns x=0 because f(0)=20 ≤ 25 (satisfies upper bound)
        assert result == 0, f"With UPPER bound constraint, should return x=0, got {result}"

    def test_solution_within_bounds(self):
        """Test cases where a solution exists within bounds and should be found"""
        
        # Case 1: Increasing function with target in range
        def increasing_func(x):
            return 2 * x + 5
        
        # Bounds: (0, 10), which means y ranges from (5, 25)
        # Target y = 15 is in range, so should find a solution
        result = search_function_with_bounds(
            func=increasing_func,
            y_target=15,
            bounds=(0, 10),
            tolerance=1e-6,
            function_type=FunctionType.INCREASING
        )
        
        assert result is not None, "Should find a solution when target is in range"
        assert abs(increasing_func(result) - 15) < 1e-5, "Solution should be accurate"
        
        # Case 2: Decreasing function with target in range
        def decreasing_func(x):
            return 20 - 2 * x
        
        # Bounds: (0, 10), which means y ranges from (20, 0)
        # Target y = 10 is in range, so should find a solution
        result = search_function_with_bounds(
            func=decreasing_func,
            y_target=10,
            bounds=(0, 10),
            tolerance=1e-6,
            function_type=FunctionType.DECREASING
        )
        
        assert result is not None, "Should find a solution for decreasing function"
        assert abs(decreasing_func(result) - 10) < 1e-5, "Solution should be accurate"
        
        # Case 3: Convex function with target in range
        def convex_func(x):
            return x**2 + 1
        
        # Target y = 5 is in range for bounds (-10, 10)
        result = search_function_with_bounds(
            func=convex_func,
            y_target=5,
            bounds=(-10, 10),
            tolerance=1e-6,
            function_type=FunctionType.CONVEX
        )
        
        assert result is not None, "Should find a solution for convex function"
        assert abs(convex_func(result) - 5) < 1e-5, "Solution should be accurate"
        
        # Case 4: Concave function with target in range
        def concave_func(x):
            return -x**2 + 10
        
        # Target y = 6 is in range for bounds (-10, 10)
        result = search_function_with_bounds(
            func=concave_func,
            y_target=6,
            bounds=(-10, 10),
            tolerance=1e-6,
            function_type=FunctionType.CONCAVE
        )
        
        assert result is not None, "Should find a solution for concave function"
        assert abs(concave_func(result) - 6) < 1e-5, "Solution should be accurate"

    def test_nonconvergent_cases(self):
        """Test cases where optimization is expected to fail to converge"""
        
        # Case 1: Function with no roots (no valid solution)
        def no_roots_func(x):
            return x**2 + 10  # Always positive, no roots
        
        # Target y = 0 has no solution
        result = search_function_with_bounds(
            func=no_roots_func,
            y_target=0,
            bounds=(-100, 100),
            tolerance=1e-10,  # Very tight tolerance
            function_type=FunctionType.CONVEX
        )
        
        assert result is None, "Should return None when no solution exists"
        
        # Case 2: Function with extremely oscillatory behavior
        def oscillatory_func(x):
            return np.sin(1000 * x)  # High frequency oscillation
        
        # Target y = 0.5 exists but might be hard to converge to with tight tolerance
        result = search_function_with_bounds(
            func=oscillatory_func,
            y_target=0.5,
            bounds=(0, 0.1),  # Small range with many oscillations
            tolerance=1e-10,  # Very tight tolerance
            function_type=FunctionType.CONCAVE  # Not actually concave, testing robustness
        )
        
        # This test allows either convergence or non-convergence for highly oscillatory functions
        # The important thing is that the function doesn't crash
        if result is not None:
            # If it converged, the solution should be reasonably accurate
            assert abs(oscillatory_func(result) - 0.5) < 1e-5, "If solution found, it should be accurate"

    def test_convergent_cases(self):
        """Test cases where optimization should reliably converge"""
        
        # Case 1: Simple linear function
        def linear_func(x):
            return 3 * x + 2
        
        # Target y = 5 has a clear solution
        result = search_function_with_bounds(
            func=linear_func,
            y_target=5,
            bounds=(-10, 10),
            tolerance=1e-6,
            function_type=FunctionType.INCREASING
        )
        
        exact_solution = (5 - 2) / 3  # x = (y - b) / m = (5 - 2) / 3 = 1
        
        assert result is not None, "Should find solution for simple linear function"
        assert abs(result - exact_solution) < 1e-6, "Solution should match analytical result"
        assert abs(linear_func(result) - 5) < 1e-6, "Function value should match target"
        
        # Case 2: Simple quadratic function
        def quadratic_func(x):
            return x**2 - 4
        
        # Target y = 0 has clear solutions at x = 2 and x = -2
        result = search_function_with_bounds(
            func=quadratic_func,
            y_target=0,
            bounds=(0, 4),  # Only looking for the positive root
            tolerance=1e-6,
            function_type=FunctionType.CONVEX
        )
        
        exact_solution = 2  # x = sqrt(4) = 2
        
        assert result is not None, "Should find solution for simple quadratic function"
        assert abs(result - exact_solution) < 1e-6, "Solution should match analytical result"
        assert abs(quadratic_func(result) - 0) < 1e-5, "Function value should match target (relaxed tolerance for numerical precision)"


class TestSearchFunctionEdgeCases:
    """Test edge cases for search functions"""
    
    def test_edge_case_validation(self):
        """Test specific edge cases that might cause optimization to fail"""
        
        # Function with behavior similar to asymptotic functions
        def steep_func(x):
            if x <= 1.001:
                return 1000  # Very large value near boundary
            return 1 / (x - 1) + 3
        
        # Target y = 4 should have a solution at x = 2
        result = search_function_with_bounds(
            func=steep_func,
            y_target=4,
            bounds=(1.01, 10),  # Away from the steep region
            tolerance=1e-4,  # Reasonable tolerance for numerical stability
            function_type=FunctionType.DECREASING
        )
        
        if result is not None:
            # If a solution is found, it should be reasonably accurate
            assert abs(steep_func(result) - 4) < 1e-3, "Solution should be reasonably accurate"
            
    def test_boundary_conditions(self):
        """Test behavior at function boundaries"""
        
        def bounded_func(x):
            return x**2
        
        # Test at the exact boundary
        result = search_function_with_bounds(
            func=bounded_func,
            y_target=4,  # Should give x = 2
            bounds=(2, 5),  # Lower bound is exactly the solution
            tolerance=1e-6,
            function_type=FunctionType.INCREASING
        )
        
        assert result is not None, "Should handle boundary cases"
        assert abs(result - 2) < 1e-5, "Should find solution at boundary"
