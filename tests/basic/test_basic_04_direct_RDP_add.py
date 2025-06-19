#!/usr/bin/env python3
"""
Tests for allocation_epsilon_RDP_add and allocation_delta_RDP_add functions
"""

import pytest
import numpy as np

from random_allocation.random_allocation_scheme.direct import (
    allocation_epsilon_RDP_add,
    allocation_delta_RDP_add,
)
from random_allocation.comparisons.definitions import PrivacyParams
from random_allocation.comparisons.structs import SchemeConfig

# Define a sample alpha orders for RDP DCO tests
ALPHA_ORDERS = [2.0, 3.0, 5.0, 10.0]

class TestDirectRDPAdd:
    def test_epsilon_RDP_add_conservative(self):
        """
        Test epsilon RDP add with conservative parameters: large sigma, small delta."""
        params = PrivacyParams(
            sigma=20.0,
            num_steps=5,
            num_selected=1,
            num_epochs=1,
            delta=1e-3
        )
        config = SchemeConfig(allocation_RDP_DCO_alpha_orders=ALPHA_ORDERS)
        eps = allocation_epsilon_RDP_add(params, config)
        assert np.isfinite(eps), f"Epsilon RDP add should be finite, got {eps}"
        assert eps > 0, f"Epsilon RDP add should be positive, got {eps}"

    def test_delta_RDP_add_conservative(self):
        """
        Test delta RDP add with conservative parameters: large sigma, small epsilon."""
        params = PrivacyParams(
            sigma=20.0,
            num_steps=10,
            num_selected=1,
            num_epochs=2,
            epsilon=0.1
        )
        config = SchemeConfig(allocation_RDP_DCO_alpha_orders=ALPHA_ORDERS)
        delta = allocation_delta_RDP_add(params, config)
        assert np.isfinite(delta), f"Delta RDP add should be finite, got {delta}"
        assert 0 < delta < 1, f"Delta RDP add should be in (0,1), got {delta}"

    def test_roundtrip_RDP_add(self):
        """
        Test that epsilon_RDP_add and delta_RDP_add are approximately inverses."""
        sigma = 5.0
        num_steps = 8
        num_epochs = 3
        target_delta = 1e-5
        params_e = PrivacyParams(
            sigma=sigma,
            num_steps=num_steps,
            num_selected=1,
            num_epochs=num_epochs,
            delta=target_delta
        )
        config = SchemeConfig(allocation_RDP_DCO_alpha_orders=ALPHA_ORDERS)
        eps = allocation_epsilon_RDP_add(params_e, config)
        # Now recover delta from eps
        params_d = PrivacyParams(
            sigma=sigma,
            num_steps=num_steps,
            num_selected=1,
            num_epochs=num_epochs,
            epsilon=eps
        )
        recovered_delta = allocation_delta_RDP_add(params_d, config)
        assert np.isfinite(recovered_delta), f"Recovered delta should be finite, got {recovered_delta}"
        # allow up to 20% relative error due to numerical differences
        assert abs(recovered_delta - target_delta) / target_delta < 0.2, \
            f"Recovered delta {recovered_delta} differs from target {target_delta} by more than 20%"

    def test_missing_delta_raises(self):
        """
        Calling epsilon RDP add without delta should raise ValueError."""
        params = PrivacyParams(
            sigma=1.0,
            num_steps=5,
            num_selected=1,
            num_epochs=1,
            # epsilon provided, delta omitted
            epsilon=0.1
        )
        config = SchemeConfig(allocation_RDP_DCO_alpha_orders=ALPHA_ORDERS)
        with pytest.raises(ValueError):
            _ = allocation_epsilon_RDP_add(params, config)

    def test_missing_epsilon_raises(self):
        """
        Calling delta RDP add without epsilon should raise ValueError."""
        params = PrivacyParams(
            sigma=1.0,
            num_steps=5,
            num_selected=1,
            num_epochs=1,
            # delta provided, epsilon omitted
            delta=1e-3
        )
        config = SchemeConfig(allocation_RDP_DCO_alpha_orders=ALPHA_ORDERS)
        with pytest.raises(ValueError):
            _ = allocation_delta_RDP_add(params, config)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
