from random_allocation.random_allocation_scheme.analytic import allocation_epsilon_analytic
from random_allocation.random_allocation_scheme.decomposition import allocation_epsilon_decomposition
from random_allocation.random_allocation_scheme.direct import allocation_epsilon_direct
from random_allocation.comparisons.definitions import PrivacyParams, SchemeConfig

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
    
    if config.direction != 'add':
        # Create config for remove direction
        remove_config = SchemeConfig(
            direction='remove',
            discretization=config.discretization,
            alpha_orders=config.allocation_direct_alpha_orders,
            print_alpha=config.print_alpha,
            delta_tolerance=config.delta_tolerance,
            epsilon_tolerance=config.epsilon_tolerance,
            epsilon_upper_bound=config.epsilon_upper_bound
        )
        
        epsilon_remove_analytic = allocation_epsilon_analytic(params=params, config=remove_config)
        epsilon_remove_decompose = allocation_epsilon_decomposition(params=params, config=remove_config)
        epsilon_remove_RDP = allocation_epsilon_direct(params=params, config=remove_config)
        epsilon_remove = min(epsilon_remove_analytic, epsilon_remove_decompose, epsilon_remove_RDP)
    
    if config.direction != 'remove':
        # Create config for add direction
        add_config = SchemeConfig(
            direction='add',
            discretization=config.discretization,
            alpha_orders=config.allocation_direct_alpha_orders,
            print_alpha=config.print_alpha,
            delta_tolerance=config.delta_tolerance,
            epsilon_tolerance=config.epsilon_tolerance,
            epsilon_upper_bound=config.epsilon_upper_bound
        )
        
        epsilon_add_analytic = allocation_epsilon_analytic(params=params, config=add_config)
        epsilon_add_decompose = allocation_epsilon_decomposition(params=params, config=add_config)
        epsilon_add_RDP = allocation_epsilon_direct(params=params, config=add_config)
        epsilon_add = min(epsilon_add_analytic, epsilon_add_decompose, epsilon_add_RDP)
    
    if config.direction == 'add':
        return epsilon_add
    if config.direction == 'remove':
        return epsilon_remove
    return max(epsilon_remove, epsilon_add) 