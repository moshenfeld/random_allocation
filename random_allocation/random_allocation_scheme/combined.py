# Standard library imports
from typing import List, Dict, Tuple, Optional, Union, Callable, Any, cast

# Third-party imports
import numpy as np

# Local application imports
from random_allocation.comparisons.definitions import PrivacyParams, SchemeConfig
from random_allocation.random_allocation_scheme.analytic import allocation_epsilon_analytic, allocation_delta_analytic
from random_allocation.random_allocation_scheme.direct import allocation_epsilon_direct, allocation_delta_direct
from random_allocation.random_allocation_scheme.RDP_DCO import allocation_epsilon_RDP_DCO, allocation_delta_RDP_DCO
from random_allocation.random_allocation_scheme.recursive import allocation_epsilon_recursive, allocation_delta_recursive
from random_allocation.random_allocation_scheme.decomposition import allocation_epsilon_decomposition, allocation_delta_decomposition
from random_allocation.random_allocation_scheme.Monte_Carlo import allocation_delta_MC

def allocation_delta_combined(params: PrivacyParams,
                             config: SchemeConfig = SchemeConfig(),
                             ) -> float:
    """
    Compute delta for the combined allocation scheme.
    
    Args:
        params: Privacy parameters (must include epsilon)
        config: Scheme configuration parameters
    
    Returns:
        Computed delta value
    """
    params.validate()
    if params.epsilon is None:
        raise ValueError("Epsilon must be provided to compute delta")
    
    return 0  # TODO: Implement combined delta method

def allocation_epsilon_combined(params: PrivacyParams,
                               config: SchemeConfig = SchemeConfig(),
                               ) -> float:
    """
    Compute epsilon for the combined allocation scheme.
    This method uses the minimum of the various allocation methods.
    
    Args:
        params: Privacy parameters (must include delta)
        config: Scheme configuration parameters
    
    Returns:
        Computed epsilon value
    """
    params.validate()
    if params.delta is None:
        raise ValueError("Delta must be provided to compute epsilon")
    
    epsilon_remove: Optional[float] = None
    if config.direction != 'add':
        # Create config for remove direction
        remove_config = SchemeConfig(
            direction='remove',
            discretization=config.discretization,
            allocation_direct_alpha_orders=config.allocation_direct_alpha_orders,
            print_alpha=config.print_alpha,
            delta_tolerance=config.delta_tolerance,
            epsilon_tolerance=config.epsilon_tolerance,
            epsilon_upper_bound=config.epsilon_upper_bound
        )
        
        # Get values and ensure they are all float
        epsilon_remove_analytic_val = allocation_epsilon_analytic(params=params, config=remove_config)
        epsilon_remove_decompose_val = allocation_epsilon_decomposition(params=params, config=remove_config)
        epsilon_remove_RDP_val = allocation_epsilon_direct(params=params, config=remove_config)
        epsilon_remove = min(
            epsilon_remove_analytic_val, 
            epsilon_remove_decompose_val, 
            epsilon_remove_RDP_val
        )
    
    epsilon_add: Optional[float] = None
    if config.direction != 'remove':
        # Create config for add direction
        add_config = SchemeConfig(
            direction='add',
            discretization=config.discretization,
            allocation_direct_alpha_orders=config.allocation_direct_alpha_orders,
            print_alpha=config.print_alpha,
            delta_tolerance=config.delta_tolerance,
            epsilon_tolerance=config.epsilon_tolerance,
            epsilon_upper_bound=config.epsilon_upper_bound
        )
        
        # Get values and ensure they are all float
        epsilon_add_analytic_val = allocation_epsilon_analytic(params=params, config=add_config)
        epsilon_add_decompose_val = allocation_epsilon_decomposition(params=params, config=add_config)
        epsilon_add_RDP_val = allocation_epsilon_direct(params=params, config=add_config)
        epsilon_add = min(
            epsilon_add_analytic_val, 
            epsilon_add_decompose_val, 
            epsilon_add_RDP_val
        )
    
    if config.direction == 'add':
        assert epsilon_add is not None, "epsilon_add should be defined in 'add' direction"
        return epsilon_add
    if config.direction == 'remove':
        assert epsilon_remove is not None, "epsilon_remove should be defined in 'remove' direction"
        return epsilon_remove
        
    # Both directions - both values should be defined at this point
    assert epsilon_add is not None, "epsilon_add should be defined"
    assert epsilon_remove is not None, "epsilon_remove should be defined" 
    return max(epsilon_remove, epsilon_add)