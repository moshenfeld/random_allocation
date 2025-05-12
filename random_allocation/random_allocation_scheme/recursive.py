# Standard library imports
from typing import Callable, Dict, Any, List, Tuple, cast

# Third-party imports
import numpy as np
from dp_accounting.pld.privacy_loss_distribution import PrivacyLossDistribution

# Local application imports
from random_allocation.comparisons.definitions import PrivacyParams, SchemeConfig
from random_allocation.comparisons.utils import search_function_with_bounds, FunctionType
from random_allocation.other_schemes.poisson import Poisson_PLD, Poisson_delta_PLD
from random_allocation.random_allocation_scheme.decomposition import allocation_delta_decomposition
from random_allocation.random_allocation_scheme.direct import allocation_delta_direct
# Type aliases
NumericFunction = Callable[[float], float]

def allocation_epsilon_recursive(params: PrivacyParams,
                                config: SchemeConfig = SchemeConfig(),
                                ) -> float:
    """
    Compute epsilon for the recursive allocation scheme.
    
    Args:
        params: Privacy parameters (must include delta)
        config: Scheme configuration parameters
    
    Returns:
        Computed epsilon value
    """
    params.validate()
    if params.delta is None:
        raise ValueError("Delta must be provided to compute epsilon")
        
    num_steps_per_round = int(np.ceil(params.num_steps/params.num_selected))
    num_rounds = int(np.ceil(params.num_steps/num_steps_per_round))
    lambda_val = 1 - (1-1.0/num_steps_per_round)**num_steps_per_round
    
    Poisson_PLD_base = Poisson_PLD(
        sigma=params.sigma, 
        num_steps=num_steps_per_round, 
        num_epochs=num_rounds*params.num_epochs, 
        sampling_prob=1.0/num_steps_per_round, 
        discretization=config.discretization, 
        direction='add'
    )
    
    if config.direction != 'add':
        epsilon_remove = np.inf
        optimization_func = lambda eps: Poisson_PLD_base.get_delta_for_epsilon(-np.log(1-lambda_val*(1-np.exp(-eps))))\
                                        *(1/(lambda_val*(np.exp(eps) -1)) - np.exp(-eps))
        gamma_result = search_function_with_bounds(
            func=optimization_func, 
            y_target=params.delta/4, 
            bounds=(1e-5, 10),
            tolerance=1e-2, 
            function_type=FunctionType.DECREASING
        )
        
        if gamma_result is not None:
            gamma = 2 * gamma_result
            sampling_prob = np.exp(gamma)/num_steps_per_round
            if sampling_prob <= np.sqrt(1/num_steps_per_round):
                Poisson_PLD_final = Poisson_PLD(
                    sigma=params.sigma, 
                    num_steps=num_steps_per_round, 
                    num_epochs=num_rounds*params.num_epochs,
                    sampling_prob=sampling_prob, 
                    discretization=config.discretization, 
                    direction='remove'
                )
                epsilon_remove = float(Poisson_PLD_final.get_epsilon_for_delta(params.delta/2))
    
    if config.direction != 'remove':
        epsilon_add = np.inf
        optimization_func = lambda eps: Poisson_PLD_base.get_delta_for_epsilon(-np.log(1-lambda_val*(1-np.exp(-eps))))\
                                        *(1/(lambda_val*(np.exp(eps) -1)) - np.exp(-eps))
        gamma_result = search_function_with_bounds(
            func=optimization_func, 
            y_target=params.delta/2, 
            bounds=(1e-5, 10),
            tolerance=1e-2, 
            function_type=FunctionType.DECREASING
        )
        
        if gamma_result is not None:
            gamma = 2 * gamma_result
            sampling_prob = np.exp(gamma)/num_steps_per_round
            if sampling_prob <= np.sqrt(1/num_steps_per_round):
                Poisson_PLD_final = Poisson_PLD(
                    sigma=params.sigma, 
                    num_steps=num_steps_per_round, 
                    num_epochs=num_rounds*params.num_epochs,
                    sampling_prob=sampling_prob, 
                    discretization=config.discretization, 
                    direction='add'
                )
                epsilon_add = float(Poisson_PLD_final.get_epsilon_for_delta(params.delta/2))
    
    if config.direction == 'add':
        assert 'epsilon_add' in locals(), "Failed to compute epsilon_add"
        return epsilon_add
    if config.direction == 'remove':
        assert 'epsilon_remove' in locals(), "Failed to compute epsilon_remove"
        return epsilon_remove
    
    # Both directions, return max of the two
    assert 'epsilon_add' in locals() and 'epsilon_remove' in locals(), "Failed to compute either epsilon_add or epsilon_remove"
    return max(epsilon_add, epsilon_remove)

def allocation_delta_recursive(params: PrivacyParams,
                               config: SchemeConfig = SchemeConfig(),
                               ) -> float:
    """
    Compute delta for the recursive allocation scheme.
    
    Args:
        params: Privacy parameters (must include epsilon)
        config: Scheme configuration parameters
    
    Returns:
        Computed delta value
    """
    params.validate()
    if params.epsilon is None:
        raise ValueError("Epsilon must be provided to compute delta")
    config_remove = SchemeConfig(
        direction='remove',
        discretization=config.discretization,
        allocation_direct_alpha_orders=config.allocation_direct_alpha_orders,
        allocation_RDP_DCO_alpha_orders=config.allocation_RDP_DCO_alpha_orders,
        Poisson_alpha_orders=config.Poisson_alpha_orders,
        print_alpha=config.print_alpha,
        delta_tolerance=config.delta_tolerance,
        epsilon_tolerance=config.epsilon_tolerance,
        epsilon_upper_bound=config.epsilon_upper_bound,
        MC_use_order_stats=config.MC_use_order_stats,
        MC_use_mean=config.MC_use_mean,
        MC_conf_level=config.MC_conf_level,
        MC_sample_size=config.MC_sample_size
    )
    config_add = SchemeConfig(
        direction='add',
        discretization=config.discretization,
        allocation_direct_alpha_orders=config.allocation_direct_alpha_orders,
        allocation_RDP_DCO_alpha_orders=config.allocation_RDP_DCO_alpha_orders,
        Poisson_alpha_orders=config.Poisson_alpha_orders,
        print_alpha=config.print_alpha,
        delta_tolerance=config.delta_tolerance,
        epsilon_tolerance=config.epsilon_tolerance,
        epsilon_upper_bound=config.epsilon_upper_bound,
        MC_use_order_stats=config.MC_use_order_stats,
        MC_use_mean=config.MC_use_mean,
        MC_conf_level=config.MC_conf_level,
        MC_sample_size=config.MC_sample_size
    )
    if config.direction != 'add':
        eps2 = min(params.epsilon/2, np.log(params.num_steps)/4)
        eta = np.exp(2*eps2)/params.num_steps
        params2 = PrivacyParams(
            sigma=params.sigma,
            num_steps=params.num_steps,
            num_selected=params.num_selected,
            num_epochs=params.num_epochs,
            epsilon=eps2,
            delta=None
        )
        delta_add_decomposition = allocation_delta_decomposition(params=params2, config=config_add)
        delta_add_direct = allocation_delta_direct(params=params2, config=config_add)
        delta_remove = Poisson_delta_PLD(params=params, config=config_remove, sampling_prob=eta) + 1/(np.exp(2*eps2)-np.exp(eps2)) * min(delta_add_decomposition, delta_add_direct)
    if config.direction != 'remove':
        if params.epsilon > 0.5:
            delta_add = allocation_delta_decomposition(params=params, config=config)
        else:
            eps2 = min(params.epsilon*2, np.log(params.num_steps)/4)
            eta = np.exp(2*eps2)/params.num_steps
            params2 = PrivacyParams(
                sigma=params.sigma,
                num_steps=params.num_steps,
                num_selected=params.num_selected,
                num_epochs=params.num_epochs,
                epsilon=eps2,
                delta=None
            )
            delta_add_decomposition = allocation_delta_decomposition(params=params2, config=config_add)
            delta_add_direct = allocation_delta_direct(params=params2, config=config_add)
            delta_add = Poisson_delta_PLD(params=params, config=config_add, sampling_prob=eta) + np.exp(eps2)/(np.exp(eps2)-1) * min(delta_add_decomposition, delta_add_direct)

    if config.direction == 'add':
        assert 'delta_add' in locals(), "Failed to compute delta_add"
        return delta_add
    if config.direction == 'remove':
        assert 'delta_remove' in locals(), "Failed to compute delta_remove"
        return delta_remove
    # Both directions, return max of the two
    assert 'delta_add' in locals(), "Failed to compute delta_add"
    assert 'delta_remove' in locals(), "Failed to compute delta_remove"
    return max(delta_add, delta_remove)
