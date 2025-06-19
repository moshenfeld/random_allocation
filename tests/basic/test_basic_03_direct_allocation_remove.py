#!/usr/bin/env python3
"""
Tests for allocation_epsilon_direct_remove and allocation_delta_direct in remove direction
"""

import pytest
import numpy as np

from random_allocation.comparisons.definitions import PrivacyParams, SchemeConfig, Direction
from random_allocation.random_allocation_scheme.direct import (
    allocation_epsilon_direct_remove,
    allocation_delta_direct,
)
from random_allocation.comparisons.structs import Verbosity  # add import

class TestAllocationDirectRemove:
    def test_epsilon_remove_conservative(self):
        """
        Test remove epsilon with conservative parameters (large sigma, small delta), single epoch.
        """
        sigma = 20.0
        num_steps = 5
        num_epochs = 1
        delta = 0.001
        alpha_orders = [2, 3, 5, 10]
        verbosity = None
        epsilon = allocation_epsilon_direct_remove(
            sigma=sigma,
            delta=delta,
            num_steps=num_steps,
            num_epochs=num_epochs,
            alpha_orders=alpha_orders,
            verbosity=verbosity,
        )
        assert np.isfinite(epsilon), f"Remove epsilon should be finite, got {epsilon}"
        assert epsilon > 0, f"Remove epsilon should be positive, got {epsilon}"

    def test_remove_delta_conservative(self):
        """
        Test remove delta with conservative parameters (large sigma, small epsilon).
        """
        sigma = 20.0
        num_steps = 5
        num_epochs = 1
        epsilon = 0.1
        params = PrivacyParams(
            sigma=sigma,
            num_steps=num_steps,
            num_selected=1,
            num_epochs=num_epochs,
            epsilon=epsilon,
        )
        config = SchemeConfig(allocation_direct_alpha_orders=[2, 3], verbosity=Verbosity.NONE)
        delta = allocation_delta_direct(
            params=params,
            config=config,
            direction=Direction.REMOVE,
        )
        assert np.isfinite(delta), f"Remove delta should be finite, got {delta}"
        assert 0 < delta < 1, f"Remove delta should be in (0,1), got {delta}"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
