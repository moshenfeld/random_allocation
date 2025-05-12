# Standard library imports
from typing import List, Dict, Tuple, Optional, Union, Callable, Any, cast

# Third-party imports
import numpy as np

# Local application imports
from random_allocation.comparisons.definitions import PrivacyParams, SchemeConfig, Direction
from random_allocation.random_allocation_scheme.analytic import allocation_epsilon_analytic, allocation_delta_analytic
from random_allocation.random_allocation_scheme.direct import allocation_epsilon_direct, allocation_delta_direct
from random_allocation.random_allocation_scheme.RDP_DCO import allocation_epsilon_RDP_DCO, allocation_delta_RDP_DCO
from random_allocation.random_allocation_scheme.recursive import allocation_epsilon_recursive, allocation_delta_recursive
from random_allocation.random_allocation_scheme.decomposition import allocation_epsilon_decomposition, allocation_delta_decomposition
from random_allocation.random_allocation_scheme.Monte_Carlo import allocation_delta_MC

def allocation_delta_combined(params: PrivacyParams,
                             config: SchemeConfig,
                             direction: Direction = Direction.BOTH,
                             ) -> float:
    """
    Compute delta for the combined allocation scheme.
    
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
    
    if direction != Direction.ADD:
        # Create config for remove direction (without direction parameter)
        remove_config = SchemeConfig(
            discretization=config.discretization,
            allocation_direct_alpha_orders=config.allocation_direct_alpha_orders,
            print_alpha=config.print_alpha,
            delta_tolerance=config.delta_tolerance,
            epsilon_tolerance=config.epsilon_tolerance,
            epsilon_upper_bound=config.epsilon_upper_bound
        )
        # Get values and ensure they are all float
        delta_remove_analytic_val = allocation_delta_analytic(params=params, config=remove_config, direction=Direction.REMOVE)
        delta_remove_decompose_val = allocation_delta_decomposition(params=params, config=remove_config, direction=Direction.REMOVE)
        delta_remove_direct_val = allocation_delta_direct(params=params, config=remove_config, direction=Direction.REMOVE)
        delta_remove_recursive_val = allocation_delta_recursive(params=params, config=remove_config, direction=Direction.REMOVE)
        delta_remove = min(
            delta_remove_analytic_val,
            delta_remove_decompose_val,
            delta_remove_direct_val,
            delta_remove_recursive_val
        )

    if direction != Direction.REMOVE:
        # Create config for add direction (without direction parameter)
        add_config = SchemeConfig(
            discretization=config.discretization,
            allocation_direct_alpha_orders=config.allocation_direct_alpha_orders,
            print_alpha=config.print_alpha,
            delta_tolerance=config.delta_tolerance,
            epsilon_tolerance=config.epsilon_tolerance,
            epsilon_upper_bound=config.epsilon_upper_bound
        )
        
        # Get values and ensure they are all float
        delta_add_analytic_val = allocation_delta_analytic(params=params, config=add_config, direction=Direction.ADD)
        delta_add_decompose_val = allocation_delta_decomposition(params=params, config=add_config, direction=Direction.ADD)
        delta_add_direct_val = allocation_delta_direct(params=params, config=add_config, direction=Direction.ADD)
        delta_add_recursive_val = allocation_delta_recursive(params=params, config=add_config, direction=Direction.ADD)
        delta_add = min(
            delta_add_analytic_val,
            delta_add_decompose_val,
            delta_add_direct_val,
            delta_add_recursive_val
        )

    if direction == Direction.ADD:
        assert delta_add is not None, "delta_add should be defined in 'add' direction"
        return delta_add
    if direction == Direction.REMOVE:
        assert delta_remove is not None, "delta_remove should be defined in 'remove' direction"
        return delta_remove
    # Both directions - both values should be defined at this point
    assert delta_add is not None, "delta_add should be defined"
    assert delta_remove is not None, "delta_remove should be defined"
    return max(delta_remove, delta_add)


def allocation_epsilon_combined(params: PrivacyParams,
                               config: SchemeConfig,
                               direction: Direction = Direction.BOTH,
                               ) -> float:
    """
    Compute epsilon for the combined allocation scheme.
    This method uses the minimum of the various allocation methods.
    
    Args:
        params: Privacy parameters (must include delta)
        config: Scheme configuration parameters
        direction: The direction of privacy. Can be ADD, REMOVE, or BOTH.
    
    Returns:
        Computed epsilon value
    """
    params.validate()
    if params.delta is None:
        raise ValueError("Delta must be provided to compute epsilon")
    
    if direction != Direction.ADD:
        # Create config for remove direction (without direction parameter)
        remove_config = SchemeConfig(
            discretization=config.discretization,
            allocation_direct_alpha_orders=config.allocation_direct_alpha_orders,
            print_alpha=config.print_alpha,
            delta_tolerance=config.delta_tolerance,
            epsilon_tolerance=config.epsilon_tolerance,
            epsilon_upper_bound=config.epsilon_upper_bound
        )
        
        # Get values and ensure they are all float
        epsilon_remove_analytic_val = allocation_epsilon_analytic(params=params, config=remove_config, direction=Direction.REMOVE)
        epsilon_remove_decompose_val = allocation_epsilon_decomposition(params=params, config=remove_config, direction=Direction.REMOVE)
        epsilon_remove_direct_val = allocation_epsilon_direct(params=params, config=remove_config, direction=Direction.REMOVE)
        epsilon_remove_recursive_val = allocation_epsilon_recursive(params=params, config=remove_config, direction=Direction.REMOVE)
        epsilon_remove = min(
            epsilon_remove_analytic_val,
            epsilon_remove_decompose_val,
            epsilon_remove_direct_val,
            epsilon_remove_recursive_val
        )

    if direction != Direction.REMOVE:
        # Create config for add direction (without direction parameter)
        add_config = SchemeConfig(
            discretization=config.discretization,
            allocation_direct_alpha_orders=config.allocation_direct_alpha_orders,
            print_alpha=config.print_alpha,
            delta_tolerance=config.delta_tolerance,
            epsilon_tolerance=config.epsilon_tolerance,
            epsilon_upper_bound=config.epsilon_upper_bound
        )
        
        # Get values and ensure they are all float
        epsilon_add_analytic_val = allocation_epsilon_analytic(params=params, config=add_config, direction=Direction.ADD)
        epsilon_add_decompose_val = allocation_epsilon_decomposition(params=params, config=add_config, direction=Direction.ADD)
        epsilon_add_direct_val = allocation_epsilon_direct(params=params, config=add_config, direction=Direction.ADD)
        epsilon_add_recursive_val = allocation_epsilon_recursive(params=params, config=add_config, direction=Direction.ADD)
        epsilon_add = min(
            epsilon_add_analytic_val,
            epsilon_add_decompose_val,
            epsilon_add_direct_val,
            epsilon_add_recursive_val
        )

    if direction == Direction.ADD:
        assert epsilon_add is not None, "epsilon_add should be defined in 'add' direction"
        return epsilon_add
    if direction == Direction.REMOVE:
        assert epsilon_remove is not None, "epsilon_remove should be defined in 'remove' direction"
        return epsilon_remove
        
    # Both directions - both values should be defined at this point
    assert epsilon_add is not None, "epsilon_add should be defined"
    assert epsilon_remove is not None, "epsilon_remove should be defined" 
    return max(epsilon_remove, epsilon_add)