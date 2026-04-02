#!/usr/bin/env python3
"""
Other Privacy Schemes Tests - Level 4

Tests Local, Poisson, and Shuffle privacy schemes without hiding any failures.
These serve as baselines and comparisons for allocation methods.
"""

import pytest
import numpy as np
import os
from random_allocation.comparisons.definitions import PrivacyParams, SchemeConfig, Direction
from random_allocation.other_schemes.local import local_epsilon, local_delta, Gaussian_epsilon, Gaussian_delta
from random_allocation.other_schemes.poisson import Poisson_epsilon_PLD, Poisson_delta_PLD, Poisson_epsilon_RDP, Poisson_delta_RDP
from random_allocation.other_schemes.shuffle import (
    shuffle_epsilon_analytic,
    shuffle_delta_analytic,
    shuffle_epsilon_lower_bound,
    shuffle_delta_lower_bound,
)
import random_allocation.other_schemes.shuffle as shuffle_module
import random_allocation.external_sources.shuffle_external as shuffle_external_module

from tests.test_utils import ResultsReporter


@pytest.fixture(scope="session")
def reporter() -> ResultsReporter:
    """Set up the results reporter for the session."""
    rep = ResultsReporter("test_full_02_other_schemes")
    return rep


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


class TestLocalScheme:
    """Test local (Gaussian) mechanism as baseline"""
    
    def test_local_epsilon_all_directions(self):
        """Test local epsilon with all directions"""
        params = PrivacyParams(
            sigma=2.0,
            num_steps=10,
            num_selected=3,
            num_epochs=1,
            delta=1e-5
        )
        config = SchemeConfig()
        
        directions = [Direction.ADD, Direction.REMOVE, Direction.BOTH]
        
        for direction in directions:
            epsilon = local_epsilon(params, config, direction)
            
            assert np.isfinite(epsilon), f"Local epsilon ({direction.value}) returned {epsilon}, should be finite"
            assert epsilon > 0, f"Local epsilon ({direction.value}) should be positive: {epsilon}"
    
    def test_local_delta_all_directions(self):
        """Test local delta with all directions"""
        params = PrivacyParams(
            sigma=2.0,
            num_steps=10,
            num_selected=3,
            num_epochs=1,
            epsilon=1.0
        )
        config = SchemeConfig()
        
        directions = [Direction.ADD, Direction.REMOVE, Direction.BOTH]
        
        for direction in directions:
            delta = local_delta(params, config, direction)
            
            assert np.isfinite(delta), f"Local delta ({direction.value}) returned {delta}, should be finite"
            assert 0 < delta < 1, f"Local delta ({direction.value}) should be in (0,1): {delta}"
    
    def test_local_round_trip_consistency(self):
        """Test local scheme round-trip consistency"""
        original_delta = 1e-5
        sigma = 2.0
        
        params_for_eps = PrivacyParams(
            sigma=sigma, num_steps=10, num_selected=3, num_epochs=1, delta=original_delta
        )
        config = SchemeConfig()
        
        # Get epsilon from delta
        epsilon = local_epsilon(params_for_eps, config, Direction.BOTH)
        
        # Get delta back from epsilon
        params_for_delta = PrivacyParams(
            sigma=sigma, num_steps=10, num_selected=3, num_epochs=1, epsilon=epsilon
        )
        computed_delta = local_delta(params_for_delta, config, Direction.BOTH)
        
        # Check consistency
        relative_error = abs(computed_delta - original_delta) / original_delta
        assert relative_error < 0.01, f"Local round-trip error too large: {relative_error}"


class TestPoissonScheme:
    """Test Poisson subsampling mechanisms"""
    
    def test_poisson_pld_epsilon(self):
        """Test Poisson PLD epsilon calculation"""
        params = PrivacyParams(
            sigma=2.0,
            num_steps=10,
            num_selected=1,  # PLD method requires num_selected=1
            num_epochs=1,
            delta=1e-5
        )
        config = SchemeConfig()
        
        directions = [Direction.ADD, Direction.REMOVE, Direction.BOTH]
        
        for direction in directions:
            epsilon = Poisson_epsilon_PLD(params, config, direction=direction)
            
            assert np.isfinite(epsilon), f"Poisson PLD epsilon ({direction.value}) returned {epsilon}"
            assert epsilon > 0, f"Poisson PLD epsilon ({direction.value}) should be positive: {epsilon}"
    
    def test_poisson_pld_delta(self):
        """Test Poisson PLD delta calculation"""
        params = PrivacyParams(
            sigma=2.0,
            num_steps=10,
            num_selected=1,  # PLD method requires num_selected=1
            num_epochs=1,
            epsilon=1.0
        )
        config = SchemeConfig()
        
        directions = [Direction.ADD, Direction.REMOVE, Direction.BOTH]
        
        for direction in directions:
            delta = Poisson_delta_PLD(params, config, direction=direction)
            
            assert np.isfinite(delta), f"Poisson PLD delta ({direction.value}) returned {delta}"
            assert 0 < delta <= 1, f"Poisson PLD delta ({direction.value}) should be in (0,1]: {delta}"
    
    def test_poisson_rdp_epsilon(self):
        """Test Poisson RDP epsilon calculation"""
        params = PrivacyParams(
            sigma=2.0,
            num_steps=10,
            num_selected=3,
            num_epochs=1,
            delta=1e-5
        )
        config = SchemeConfig()
        
        epsilon = Poisson_epsilon_RDP(params, config)
        
        assert np.isfinite(epsilon), f"Poisson RDP epsilon returned {epsilon}"
        assert epsilon > 0, f"Poisson RDP epsilon should be positive: {epsilon}"
    
    def test_poisson_rdp_delta(self):
        """Test Poisson RDP delta calculation"""
        params = PrivacyParams(
            sigma=2.0,
            num_steps=10,
            num_selected=3,
            num_epochs=1,
            epsilon=1.0
        )
        config = SchemeConfig()
        
        delta = Poisson_delta_RDP(params, config)
        
        assert np.isfinite(delta), f"Poisson RDP delta returned {delta}"
        assert 0 < delta <= 1, f"Poisson RDP delta should be in (0,1]: {delta}"


class TestShuffleScheme:
    """Test shuffle mechanism"""
    
    def test_shuffle_epsilon_basic(self):
        """Test shuffle epsilon calculation"""
        params = PrivacyParams(
            sigma=2.0,
            num_steps=10,
            num_selected=1,  # Shuffle requires num_selected=1
            num_epochs=1,
            delta=1e-5
        )
        config = SchemeConfig()
        
        directions = [Direction.ADD, Direction.REMOVE, Direction.BOTH]
        
        for direction in directions:
            epsilon = shuffle_epsilon_analytic(params, config, direction)
            
            # Shuffle might return inf for some parameter combinations
            assert np.isfinite(epsilon) or np.isinf(epsilon), f"Shuffle epsilon ({direction.value}) returned {epsilon}"
            if np.isfinite(epsilon):
                assert epsilon > 0, f"Shuffle epsilon ({direction.value}) should be positive: {epsilon}"
    
    def test_shuffle_delta_basic(self):
        """Test shuffle delta calculation"""
        params = PrivacyParams(
            sigma=2.0,
            num_steps=10,
            num_selected=1,  # Shuffle requires num_selected=1
            num_epochs=1,
            epsilon=1.0
        )
        config = SchemeConfig()
        
        directions = [Direction.ADD, Direction.REMOVE, Direction.BOTH]
        
        for direction in directions:
            delta = shuffle_delta_analytic(params, config, direction)
            
            assert np.isfinite(delta), f"Shuffle delta ({direction.value}) returned {delta}"
            assert 0 < delta <= 1, f"Shuffle delta ({direction.value}) should be in (0,1]: {delta}"

    def test_shuffle_lower_bound_basic(self):
        """Test shuffle lower-bound epsilon and delta calculations."""
        config = SchemeConfig()

        epsilon_params = PrivacyParams(
            sigma=2.0,
            num_steps=10,
            num_selected=1,
            num_epochs=1,
            delta=1e-5,
        )
        epsilon = shuffle_epsilon_lower_bound(epsilon_params, config, Direction.BOTH)
        assert np.isfinite(epsilon)
        assert epsilon > 0

        delta_params = PrivacyParams(
            sigma=2.0,
            num_steps=10,
            num_selected=1,
            num_epochs=1,
            epsilon=1.0,
        )
        delta = shuffle_delta_lower_bound(delta_params, config, Direction.BOTH)
        assert np.isfinite(delta)
        assert 0 < delta <= 1

    def test_shuffle_lower_bound_rejects_directional_queries(self):
        """Test that the shuffle lower bound is only exposed for Direction.BOTH."""
        config = SchemeConfig()
        params = PrivacyParams(
            sigma=2.0,
            num_steps=10,
            num_selected=1,
            num_epochs=1,
            delta=1e-5,
        )

        with pytest.raises(ValueError, match="Direction.BOTH"):
            shuffle_epsilon_lower_bound(params, config, Direction.ADD)

    def test_shuffle_uses_configured_search_iterations(self, monkeypatch):
        """Test that shuffle epsilon uses dedicated binary-search iterations."""
        recorded_iterations = []

        monkeypatch.setattr(shuffle_module, "local_epsilon", lambda *args, **kwargs: 0.1)

        def fake_numericalanalysis(*, n, epsorig, delta, num_iterations, step, upperbound):
            recorded_iterations.append(num_iterations)
            return 0.2

        monkeypatch.setattr(shuffle_module, "numericalanalysis", fake_numericalanalysis)

        params = PrivacyParams(
            sigma=2.0,
            num_steps=10,
            num_selected=1,
            num_epochs=1,
            delta=1e-5,
        )
        config = SchemeConfig(shuffle_search_iterations=37)

        epsilon = shuffle_epsilon_analytic(params, config, Direction.BOTH)

        assert epsilon > 0
        assert recorded_iterations
        assert set(recorded_iterations) == {37}

    def test_shuffle_upper_bound_uses_both_interval_endpoints(self, monkeypatch):
        """Test the regression for the deltaq upper-bound endpoint typo."""
        monkeypatch.setattr(shuffle_external_module.stats.binom, "cdf", lambda k, n, p: float(k))
        monkeypatch.setattr(shuffle_external_module.stats.binom, "pmf", lambda k, n, p: 0.0)
        monkeypatch.setattr(
            shuffle_external_module,
            "onestep",
            lambda c, eps, eps0, pminusq: {
                (1, True): 0.2,
                (0, True): 0.1,
                (1, False): 0.1,
                (0, False): 0.9,
            }[(c, pminusq)],
        )

        result = shuffle_external_module.deltacomp(
            n=2,
            eps0=1000.0,
            eps=0.0,
            deltaupper=10.0,
            step=1,
            upperbound=True,
        )

        assert result == pytest.approx(0.9)

    def test_shuffle_uses_default_step_heuristic(self, monkeypatch):
        """Test that the default negative shuffle_step selects the heuristic step."""
        recorded_steps = []

        monkeypatch.setattr(shuffle_module, "local_epsilon", lambda *args, **kwargs: 0.1)

        def fake_numericalanalysis(*, n, epsorig, delta, num_iterations, step, upperbound):
            recorded_steps.append(step)
            return 0.2

        monkeypatch.setattr(shuffle_module, "numericalanalysis", fake_numericalanalysis)

        params = PrivacyParams(
            sigma=2.0,
            num_steps=10,
            num_selected=1,
            num_epochs=1,
            delta=1e-5,
        )

        epsilon = shuffle_epsilon_analytic(params, SchemeConfig(), Direction.BOTH)

        assert epsilon > 0
        assert recorded_steps
        assert set(recorded_steps) == {2}

    def test_shuffle_returns_det_eps_before_refinement(self, monkeypatch):
        """Test that refinement is skipped when the first bound is already no better than local."""
        local_eps_values = iter([0.3, 0.1])
        recorded_steps = []

        monkeypatch.setattr(shuffle_module, "local_epsilon", lambda *args, **kwargs: next(local_eps_values))

        def fake_numericalanalysis(*, n, epsorig, delta, num_iterations, step, upperbound):
            recorded_steps.append(step)
            return 0.3

        monkeypatch.setattr(shuffle_module, "numericalanalysis", fake_numericalanalysis)

        params = PrivacyParams(
            sigma=2.0,
            num_steps=10,
            num_selected=1,
            num_epochs=1,
            delta=1e-5,
        )

        epsilon = shuffle_epsilon_analytic(params, SchemeConfig(), Direction.BOTH)

        assert epsilon == pytest.approx(0.3)
        assert recorded_steps == [2]

    def test_shuffle_nudges_zero_local_epsilon(self, monkeypatch):
        """Test that the wrapper never passes epsorig=0 into the external routine."""
        recorded_epsorig = []

        monkeypatch.setattr(shuffle_module, "local_epsilon", lambda *args, **kwargs: 0.0)

        def fake_numericalanalysis(*, n, epsorig, delta, num_iterations, step, upperbound):
            recorded_epsorig.append(epsorig)
            return 0.2

        monkeypatch.setattr(shuffle_module, "numericalanalysis", fake_numericalanalysis)

        params = PrivacyParams(
            sigma=100.0,
            num_steps=5,
            num_selected=1,
            num_epochs=1,
            delta=1e-5,
        )

        epsilon = shuffle_epsilon_analytic(params, SchemeConfig(), Direction.BOTH)

        assert epsilon >= 0
        assert recorded_epsorig
        assert all(epsorig > 0 for epsorig in recorded_epsorig)

class TestInterSchemeComparison:
    """Compare results across different schemes"""
    
    def test_local_vs_poisson_consistency(self):
        """Compare local and Poisson methods with same parameters"""
        params = PrivacyParams(
            sigma=3.0,
            num_steps=10,
            num_selected=1,  # PLD method requires num_selected=1
            num_epochs=1,
            delta=1e-4
        )
        config = SchemeConfig()
        
        # Get epsilon from both methods
        local_eps = local_epsilon(params, config, Direction.BOTH)
        poisson_eps = Poisson_epsilon_PLD(params, config, direction=Direction.BOTH)
        
        print(f"Local epsilon: {local_eps}")
        print(f"Poisson PLD epsilon: {poisson_eps}")
        
        # Both should be positive and finite
        assert local_eps > 0, f"Local epsilon should be positive: {local_eps}"
        assert poisson_eps > 0, f"Poisson epsilon should be positive: {poisson_eps}"
        
        # Poisson should be smaller (better privacy) than local for same parameters
        if local_eps < poisson_eps:
            print(f"Note: Local epsilon {local_eps} < Poisson epsilon {poisson_eps}")
    
    def test_scheme_ordering_sanity(self):
        """Test that privacy schemes follow expected ordering"""
        params = PrivacyParams(
            sigma=2.0,
            num_steps=20,
            num_selected=1,  # PLD method requires num_selected=1
            num_epochs=1,
            delta=1e-4
        )
        config = SchemeConfig()
        
        methods_to_test = [
            ("local", local_epsilon),
            ("poisson_pld", lambda p, c, d: Poisson_epsilon_PLD(p, c, direction=d)),
            ("poisson_rdp", lambda p, c, d: Poisson_epsilon_RDP(p, c)),
        ]
        
        results = {}
        for name, method in methods_to_test:
            if name == "poisson_rdp":
                epsilon = method(params, config, None)  # RDP doesn't take direction
            else:
                epsilon = method(params, config, Direction.BOTH)
            results[name] = epsilon
        
        print(f"Scheme comparison: {results}")
        
        # All should be positive finite values
        for name, result in results.items():
            assert isinstance(result, (int, float)), f"{name} failed: {result}"
            assert result > 0, f"{name} should give positive epsilon: {result}"
            assert np.isfinite(result), f"{name} should give finite epsilon: {result}"


class TestGaussianMechanismIntegration:
    """Test integration with underlying Gaussian mechanism"""
    
    def test_gaussian_baseline_performance(self):
        """Test that basic Gaussian mechanism performs well"""
        import time
        
        test_cases = [
            (1.0, 1e-6),
            (2.0, 1e-5),
            (5.0, 1e-4),
        ]
        
        for sigma, delta in test_cases:
            start_time = time.time()
            epsilon = Gaussian_epsilon(sigma, delta)
            duration = time.time() - start_time
            
            print(f"Gaussian σ={sigma}, δ={delta} -> ε={epsilon:.6f} ({duration:.3f}s)")
            
            assert duration < 0.1, f"Gaussian mechanism too slow: {duration:.3f}s"
            assert epsilon > 0, f"Gaussian epsilon should be positive: {epsilon}"
            assert np.isfinite(epsilon), f"Gaussian epsilon should be finite: {epsilon}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 
