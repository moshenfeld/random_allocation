# from functools import cache
import numpy as np

from random_allocation.comparisons.utils import search_function_with_bounds, FunctionType
from random_allocation.other_schemes.poisson import Poisson_PLD
from random_allocation.comparisons.definitions import PrivacyParams, SchemeConfig

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
        optimization_func = lambda eps: Poisson_PLD_base.get_delta_for_epsilon(-np.log(1-lambda_val*(1-np.exp(-eps))))\
                                        *(1/(lambda_val*(np.exp(eps) -1)) - np.exp(-eps))
        gamma = 2*search_function_with_bounds(
            func=optimization_func, 
            y_target=params.delta/4, 
            bounds=(1e-5, 10),
            tolerance=1e-2, 
            function_type=FunctionType.DECREASING
        )
        
        sampling_prob = np.exp(gamma)/num_steps_per_round
        if sampling_prob > np.sqrt(1/num_steps_per_round):
            epsilon_remove = np.inf
        else:
            Poisson_PLD_final = Poisson_PLD(
                sigma=params.sigma, 
                num_steps=num_steps_per_round, 
                num_epochs=num_rounds*params.num_epochs,
                sampling_prob=sampling_prob, 
                discretization=config.discretization, 
                direction='remove'
            )
            epsilon_remove = Poisson_PLD_final.get_epsilon_for_delta(params.delta/2)
    
    if config.direction != 'remove':
        optimization_func = lambda eps: Poisson_PLD_base.get_delta_for_epsilon(-np.log(1-lambda_val*(1-np.exp(-eps))))\
                                        *(1/(lambda_val*(np.exp(eps) -1)) - np.exp(-eps))
        gamma = 2*search_function_with_bounds(
            func=optimization_func, 
            y_target=params.delta/2, 
            bounds=(1e-5, 10),
            tolerance=1e-2, 
            function_type=FunctionType.DECREASING
        )
        
        sampling_prob = np.exp(gamma)/num_steps_per_round
        if sampling_prob > np.sqrt(1/num_steps_per_round):
            epsilon_add = np.inf
        else:
            Poisson_PLD_final = Poisson_PLD(
                sigma=params.sigma, 
                num_steps=num_steps_per_round, 
                num_epochs=num_rounds*params.num_epochs,
                sampling_prob=sampling_prob, 
                discretization=config.discretization, 
                direction='add'
            )
            epsilon_add = Poisson_PLD_final.get_epsilon_for_delta(params.delta/2)
    
    if config.direction == 'add':
        return epsilon_add
    if config.direction == 'remove':
        return epsilon_remove
    return max(epsilon_remove, epsilon_add)

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
        return allocation_epsilon_recursive(params=params_copy, config=config)
    
    return search_function_with_bounds(
        func=optimization_func, 
        y_target=params.epsilon, 
        bounds=(config.delta_tolerance, 1-config.delta_tolerance),
        tolerance=config.delta_tolerance, 
        function_type=FunctionType.DECREASING
    )