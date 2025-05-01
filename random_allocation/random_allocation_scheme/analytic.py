# from functools import cache
import numpy as np

from random_allocation.comparisons.utils import search_function_with_bounds, FunctionType
from random_allocation.other_schemes.local import local_epsilon, FunctionType
from random_allocation.other_schemes.poisson import Poisson_epsilon_PLD
from random_allocation.comparisons.definitions import PrivacyParams, SchemeConfig

def sampling_prob_from_sigma(sigma: float,
                             delta: float,
                             num_steps: int,
                             num_selected: int,
                             local_delta: float,
                             ) -> float:
    params = PrivacyParams(sigma=sigma, delta=local_delta, num_steps=num_steps, num_selected=num_selected, num_epochs=1)
    local_epsilon_val = local_epsilon(params=params, config=SchemeConfig())
    if local_epsilon_val is None:
        return 1.0
    gamma = np.cosh(local_epsilon_val)*np.sqrt(2*num_selected*np.log(num_selected/delta)/num_steps)
    if gamma > 1 - num_selected/num_steps:
        return 1.0
    return np.clip(num_selected/(num_steps*(1.0-gamma)), 0, 1)

def allocation_epsilon_analytic(params: PrivacyParams,
                                config: SchemeConfig = SchemeConfig(),
                                ) -> float:
    """
    Compute epsilon for the analytic allocation scheme.
    
    Args:
        params: Privacy parameters
        config: Scheme configuration parameters
    
    Returns:
        Computed epsilon value
    """
    params.validate()
    if params.delta is None:
        raise ValueError("Delta must be provided to compute epsilon")
        
    local_delta_split = 0.99
    Poisson_delta_split = (1-local_delta_split)/2
    large_sampling_prob_delta_split = (1-local_delta_split)/2
    
    local_delta = params.delta*local_delta_split/(params.num_steps*params.num_epochs)
    Poisson_delta = params.delta*Poisson_delta_split
    large_sampling_prob_delta = params.delta*large_sampling_prob_delta_split/params.num_epochs
    
    sampling_prob = sampling_prob_from_sigma(
        sigma=params.sigma, 
        delta=large_sampling_prob_delta, 
        num_steps=params.num_steps,
        num_selected=params.num_selected, 
        local_delta=local_delta
    )
    
    if sampling_prob > np.sqrt(params.num_selected/params.num_steps):
        return np.inf
        
    Poisson_params = PrivacyParams(
        sigma=params.sigma, 
        delta=Poisson_delta, 
        num_steps=params.num_steps, 
        num_selected=params.num_selected, 
        num_epochs=params.num_epochs
    )
    epsilon = Poisson_epsilon_PLD(params=Poisson_params,
        config=config,
        sampling_prob=sampling_prob
    )
    
    return epsilon

def allocation_delta_analytic(params: PrivacyParams,
                              config: SchemeConfig = SchemeConfig(),
                              ) -> float:
    """
    Compute delta for the analytic allocation scheme.
    
    Args:
        params: Privacy parameters
        config: Scheme configuration parameters
        
    Returns:
        Computed delta value
    """
    params.validate()
    if params.epsilon is None:
        raise ValueError("Epsilon must be provided to compute delta")
    
    # Create a copy of params with epsilon=None to use in optimization function
    params_copy = PrivacyParams(
        sigma=params.sigma,
        num_steps=params.num_steps,
        num_selected=params.num_selected,
        num_epochs=params.num_epochs,
        epsilon=None,
        delta=None  # This will be set by the optimization function
    )
    
    def optimization_func(delta):
        params_copy.delta = delta
        return allocation_epsilon_analytic(params=params_copy, config=config)
    
    return search_function_with_bounds(
        func=optimization_func, 
        y_target=params.epsilon, 
        bounds=(config.delta_tolerance, 1-config.delta_tolerance),
        tolerance=config.delta_tolerance, 
        function_type=FunctionType.DECREASING
    )