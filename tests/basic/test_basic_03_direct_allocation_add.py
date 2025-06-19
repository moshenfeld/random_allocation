#!/usr/bin/env python3
"""
Tests for allocation_epsilon_direct_add and allocation_delta_direct_add functions
"""

import pytest
import numpy as np

from random_allocation.random_allocation_scheme.direct import (
    allocation_epsilon_direct_add,
    allocation_delta_direct_add,
)

class TestAllocationDirectAdd:
    def test_epsilon_conservative(self):
        """
        Test epsilon computation with conservative parameters (large sigma, small delta).
        """
        sigma = 20.0
        num_steps = 5
        num_epochs = 1
        delta = 0.001
        epsilon = allocation_epsilon_direct_add(
            sigma=sigma,
            delta=delta,
            num_steps=num_steps,
            num_epochs=num_epochs,
        )
        assert np.isfinite(epsilon), f"Epsilon should be finite, got {epsilon}"
        assert epsilon > 0, f"Epsilon should be positive, got {epsilon}"

    def test_delta_conservative(self):
        """
        Test delta computation with conservative parameters (large sigma, small epsilon).
        """
        sigma = 20.0
        num_steps = 10
        num_epochs = 2
        epsilon = 0.1
        delta = allocation_delta_direct_add(
            sigma=sigma,
            epsilon=epsilon,
            num_steps=num_steps,
            num_epochs=num_epochs,
        )
        assert np.isfinite(delta), f"Delta should be finite, got {delta}"
        assert 0 < delta < 1, f"Delta should be in (0,1), got {delta}"

    def test_roundtrip_delta_epsilon(self):
        """
        Test that epsilon and delta are approximately inverses for a given scheme.
        """
        sigma = 5.0
        num_steps = 10
        num_epochs = 3
        target_delta = 1e-5
        epsilon = allocation_epsilon_direct_add(
            sigma=sigma,
            delta=target_delta,
            num_steps=num_steps,
            num_epochs=num_epochs,
        )
        recovered_delta = allocation_delta_direct_add(
            sigma=sigma,
            epsilon=epsilon,
            num_steps=num_steps,
            num_epochs=num_epochs,
        )
        assert np.isfinite(recovered_delta), f"Recovered delta should be finite, got {recovered_delta}"
        # Allow ~10% relative error in the inverse calculation
        assert abs(recovered_delta - target_delta) / target_delta < 0.1, \
            f"Recovered delta {recovered_delta} differs from target {target_delta} by more than 10%"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
