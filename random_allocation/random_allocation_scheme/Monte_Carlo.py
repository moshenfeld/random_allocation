# Standard library imports
from typing import Optional, Union, Callable, Dict, Any, List, Tuple, cast, Literal

# Third-party imports
import numpy as np

# Local application imports
from random_allocation.random_allocation_scheme.Monte_Carlo_external import *
from random_allocation.comparisons.definitions import PrivacyParams, SchemeConfig, Direction

def Monte_Carlo_estimation(params: PrivacyParams,
                           config: SchemeConfig,
                           adjacency_type: AdjacencyType,
                           ) -> Dict[str, float]:
    """
    Estimate delta using Monte Carlo simulation.
    
    Args:
        params: Privacy parameters
        config: Scheme configuration
        adjacency_type: Type of adjacency (ADD or REMOVE)
    
    Returns:
        Estimated delta value
    """
    bnb_accountant = BnBAccountant()
    if config.MC_use_order_stats:
        # Define order_stats_encoding with a flexible type
        order_stats_encoding: Tuple[int, ...]
        if params.num_steps < 100:
            order_stats_encoding = (1, params.num_steps, 1)
        elif params.num_steps < 500:
            order_stats_encoding = (1, 100, 1, 100, params.num_steps, 10)
        elif params.num_steps < 1000:
            order_stats_encoding = (1, 100, 1, 100, 500, 10, 500, params.num_steps, 50)
        else:
            order_stats_encoding = (1, 100, 1, 100, 500, 10, 500, 1000, 50, 1000, params.num_steps, 100)
        order_stats_seq = get_order_stats_seq_from_encoding(order_stats_encoding, params.num_steps)
        delta_estimate = bnb_accountant.estimate_order_stats_deltas(
            params.sigma, 
            [params.epsilon], 
            params.num_steps, 
            config.MC_sample_size, 
            order_stats_seq,
            params.num_epochs, 
            adjacency_type
        )[0]
    else:
        delta_estimate = bnb_accountant.estimate_deltas(
            params.sigma, 
            [params.epsilon], 
            params.num_steps, 
            config.MC_sample_size, 
            params.num_epochs, 
            adjacency_type, 
            use_importance_sampling=True
        )[0]
    return {'mean': float(delta_estimate.mean), 
            'high prob': float(delta_estimate.get_upper_confidence_bound(1-config.MC_conf_level))}

def allocation_delta_MC(params: PrivacyParams, 
                      config: SchemeConfig,
                      direction: Direction = Direction.BOTH) -> float:
    """
    Compute delta using Monte Carlo simulation for the allocation scheme.
    
    Args:
        params: Privacy parameters (must include epsilon)
        config: Scheme configuration parameters
        direction: The direction of privacy. Can be ADD, REMOVE, or BOTH.
    
    Returns:
        Computed delta value
    """
    params.validate()
    if params.epsilon is None:
        raise ValueError("Epsilon must be provided to compute delta")
    
    assert params.num_selected == 1, "Monte Carlo only supports num_selected=1"
    
    # Variables that will be defined conditionally
    delta_add: float  # type annotation without initialization
    delta_remove: float  # type annotation without initialization
    
    return_field = 'mean' if config.MC_use_mean else 'high_prob'
    if direction != Direction.ADD:
        delta_remove = Monte_Carlo_estimation(params, config, AdjacencyType.REMOVE)[return_field]
    if direction != Direction.REMOVE:
        delta_add = Monte_Carlo_estimation(params, config, AdjacencyType.ADD)[return_field]

    if direction == Direction.ADD:
        assert 'delta_add' in locals(), "Failed to compute delta_add"
        return delta_add
    if direction == Direction.REMOVE:
        assert 'delta_remove' in locals(), "Failed to compute delta_remove"
        return delta_remove
    
    # Both directions, return max
    assert 'delta_add' in locals() and 'delta_remove' in locals(), "Failed to compute either delta_add or delta_remove"
    return float(max(delta_add, delta_remove))
