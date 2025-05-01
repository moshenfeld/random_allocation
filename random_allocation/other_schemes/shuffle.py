# from functools import cache
import numpy as np

from random_allocation.comparisons.utils import search_function_with_bounds, FunctionType
from random_allocation.other_schemes.shuffle_external import numericalanalysis
from random_allocation.comparisons.definitions import PrivacyParams, SchemeConfig
from random_allocation.other_schemes.local import local_epsilon

# @cache
def shuffle_epsilon_analytic(params: PrivacyParams,
                             config: SchemeConfig = SchemeConfig(),
                             step: float = 100,
                             ) -> float:
    """
    Calculate the epsilon value for the shuffle scheme.
    
    Parameters:
    - params: Privacy parameters
    - config: Scheme configuration parameters
    - step: Step size for numerical analysis
    """
    params.validate()
    if params.delta is None:
        raise ValueError("Delta must be provided to compute epsilon")
    
    if params.num_epochs > 1 or params.num_selected > 1:
        raise ValueError('Shuffle method only supports num_epochs=1 and num_selected=1')
    
    delta_split = 0.05
    
    # Create temporary params for local_epsilon
    temp_params = PrivacyParams(
        sigma=params.sigma,
        delta=params.delta,
        epsilon=None,
        num_steps=params.num_steps,
        num_selected=params.num_selected,
        num_epochs=params.num_epochs
    )
    det_eps = local_epsilon(params=temp_params, config=config)
    
    # Create params for the local delta
    local_delta = params.delta*delta_split/(2*params.num_steps*(np.exp(2)+1)*(1+np.exp(2)/2))
    local_params = PrivacyParams(
        sigma=params.sigma,
        delta=local_delta,
        epsilon=None,
        num_steps=params.num_steps,
        num_selected=1,
        num_epochs=1
    )
    
    local_epsilon_val = local_epsilon(params=local_params, config=config)
    if local_epsilon_val is None or local_epsilon_val > 10:
        return det_eps
    
    epsilon = numericalanalysis(
        n=params.num_steps, 
        epsorig=local_epsilon_val, 
        delta=params.delta*(1-delta_split), 
        num_iterations=params.num_epochs,
        step=step, 
        upperbound=True
    )
    
    for _ in range(5):
        local_delta = params.delta/(2*params.num_steps*(np.exp(epsilon)+1)*(1+np.exp(local_epsilon_val)/2))
        local_params.delta = local_delta
        local_epsilon_val = local_epsilon(params=local_params, config=config)
        
        epsilon = numericalanalysis(
            n=params.num_steps, 
            epsorig=local_epsilon_val, 
            delta=params.delta*(1-delta_split),
            num_iterations=params.num_epochs, 
            step=step, 
            upperbound=True
        )
        
        delta_bnd = params.delta*(1-delta_split)+local_delta*params.num_steps*(np.exp(epsilon)+1)*(1+np.exp(local_epsilon_val)/2)
        if delta_bnd < params.delta:
            break
    
    if epsilon > det_eps:
        return det_eps
    
    return epsilon

def shuffle_delta_analytic(params: PrivacyParams,
                           config: SchemeConfig = SchemeConfig(),
                           step: float = 100,
                           ) -> float:
    """
    Calculate the delta value for the shuffle scheme.
    
    Parameters:
    - params: Privacy parameters
    - config: Scheme configuration parameters
    - step: Step size for numerical analysis
    """
    params.validate()
    if params.epsilon is None:
        raise ValueError("Epsilon must be provided to compute delta")
    
    if params.num_epochs > 1 or params.num_selected > 1:
        raise ValueError('Shuffle method only supports num_epochs=1 and num_selected=1')
    
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
        return shuffle_epsilon_analytic(params=params_copy, config=config, step=step)
    
    return search_function_with_bounds(
        func=optimization_func, 
        y_target=params.epsilon, 
        bounds=(config.delta_tolerance, 1-config.delta_tolerance),
        tolerance=config.delta_tolerance, 
        function_type=FunctionType.DECREASING
    )