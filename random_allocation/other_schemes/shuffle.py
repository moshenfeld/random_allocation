# Standard library imports
from dataclasses import replace
from typing import Optional, Union, Callable, Dict, Any

# Third-party imports
import numpy as np

# Local application imports
from random_allocation.comparisons.definitions import PrivacyParams, SchemeConfig, Direction
from random_allocation.comparisons.utils import search_function_with_bounds, FunctionType
from random_allocation.external_sources.Monte_Carlo_external import ShuffleAccountant
from random_allocation.external_sources.shuffle_external import numericalanalysis
from random_allocation.other_schemes.local import local_epsilon

def _effective_shuffle_step(num_steps: int, configured_step: float) -> int:
    if configured_step < 0:
        return max(1, int(np.floor(np.sqrt(num_steps / 2.0))))
    if configured_step == 0:
        raise ValueError("shuffle_step must be positive, or negative to use the default heuristic")
    return max(1, int(np.floor(configured_step)))


def _strictly_positive_local_epsilon(local_epsilon_val: float) -> float:
    return max(float(local_epsilon_val), float(np.finfo(float).eps))


def shuffle_epsilon_analytic(params: PrivacyParams,
                             config: SchemeConfig,
                             direction: Direction = Direction.BOTH,
                             ) -> float:
    """
    Calculate the epsilon value for the shuffle scheme.
    
    Parameters:
    - params: Privacy parameters
    - config: Scheme configuration parameters
    - direction: The direction of privacy. Can be ADD, REMOVE, or BOTH.
    """
    if params.delta is None:
        raise ValueError("Delta must be provided to compute epsilon")
    
    if params.num_epochs > 1 or params.num_selected > 1:
        raise ValueError('Shuffle method only supports num_epochs=1 and num_selected=1')
    
    search_iterations = max(int(config.shuffle_search_iterations), 1)

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
    det_eps = local_epsilon(params=temp_params, config=config, direction=direction)
    if params.num_steps <= 1:
        return float(det_eps) if det_eps is not None else float('inf')

    shuffle_step = _effective_shuffle_step(params.num_steps, config.shuffle_step)
    
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
    
    local_epsilon_val = local_epsilon(params=local_params, config=config, direction=direction)
    if local_epsilon_val is None or local_epsilon_val > 10:
        return float(det_eps) if det_eps is not None else float('inf')
    local_epsilon_val = _strictly_positive_local_epsilon(local_epsilon_val)
    
    epsilon = numericalanalysis(
        n=params.num_steps, 
        epsorig=local_epsilon_val, 
        delta=params.delta*(1-delta_split), 
        num_iterations=search_iterations,
        step=shuffle_step,
        upperbound=True
    )
    if det_eps is not None and epsilon >= det_eps:
        return float(det_eps)
    
    for _ in range(5):
        local_delta = params.delta/(2*params.num_steps*(np.exp(epsilon)+1)*(1+np.exp(local_epsilon_val)/2))
        local_params.delta = local_delta
        local_epsilon_val = local_epsilon(params=local_params, config=config, direction=direction)
        if local_epsilon_val is None:
            local_epsilon_val = float('inf')  # Use infinity for None values
        else:
            local_epsilon_val = _strictly_positive_local_epsilon(local_epsilon_val)
            
        epsilon = numericalanalysis(
            n=params.num_steps, 
            epsorig=local_epsilon_val, 
            delta=params.delta*(1-delta_split),
            num_iterations=search_iterations, 
            step=shuffle_step,
            upperbound=True
        )
        
        delta_bnd = params.delta*(1-delta_split)+local_delta*params.num_steps*(np.exp(epsilon)+1)*(1+np.exp(local_epsilon_val)/2)
        if delta_bnd < params.delta:
            break
    
    if epsilon > det_eps and det_eps is not None:
        return float(det_eps)
    
    # Return epsilon but ensure it's a float
    return float(epsilon)

def shuffle_delta_analytic(params: PrivacyParams,
                           config: SchemeConfig,
                           direction: Direction = Direction.BOTH,
                           ) -> float:
    """
    Calculate the delta value for the shuffle scheme.
    
    Parameters:
    - params: Privacy parameters
    - config: Scheme configuration parameters
    - direction: The direction of privacy. Can be ADD, REMOVE, or BOTH.
    """
    if params.epsilon is None:
        raise ValueError("Epsilon must be provided to compute delta")
    
    if params.num_epochs > 1 or params.num_selected > 1:
        raise ValueError('Shuffle method only supports num_epochs=1 and num_selected=1')

    delta_search_tolerance = max(config.delta_tolerance, 1e-6)
    delta_search_config = replace(
        config,
        shuffle_search_iterations=min(max(int(config.shuffle_search_iterations), 1), 20),
    )
        
    result = search_function_with_bounds(
        func=lambda delta: shuffle_epsilon_analytic(params=PrivacyParams(
            sigma=params.sigma,
            num_steps=params.num_steps,
            num_selected=params.num_selected,
            num_epochs=params.num_epochs,
            epsilon=None,
            delta=delta
        ), config=delta_search_config, direction=direction),
        y_target=params.epsilon, 
        bounds=(config.delta_tolerance, 1-config.delta_tolerance),
        tolerance=delta_search_tolerance,
        function_type=FunctionType.DECREASING
    )
    
    # Handle the case where search_function_with_bounds returns None
    return 1.0 if result is None else float(result)


def shuffle_epsilon_lower_bound(
    params: PrivacyParams,
    config: SchemeConfig,
    direction: Direction = Direction.BOTH,
) -> float:
    """
    Compute the lower bound on epsilon for the shuffle scheme.

    This lower bound is only defined for the worst-case shuffled mechanism, so
    it supports Direction.BOTH but not directional ADD/REMOVE queries.
    """
    if direction != Direction.BOTH:
        raise ValueError("Shuffle lower bound only supports Direction.BOTH")
    if params.delta is None:
        raise ValueError("Delta must be provided to compute epsilon")
    if params.num_selected > 1:
        raise ValueError('Shuffle lower bound only supports num_selected=1')

    accountant = ShuffleAccountant()
    result = accountant.get_epsilons(
        params.sigma,
        (params.delta,),
        params.num_steps,
        params.num_epochs,
    )[0]
    return float(result)


def shuffle_delta_lower_bound(
    params: PrivacyParams,
    config: SchemeConfig,
    direction: Direction = Direction.BOTH,
) -> float:
    """
    Compute the lower bound on delta for the shuffle scheme.

    This lower bound is only defined for the worst-case shuffled mechanism, so
    it supports Direction.BOTH but not directional ADD/REMOVE queries.
    """
    if direction != Direction.BOTH:
        raise ValueError("Shuffle lower bound only supports Direction.BOTH")
    if params.epsilon is None:
        raise ValueError("Epsilon must be provided to compute delta")
    if params.num_selected > 1:
        raise ValueError('Shuffle lower bound only supports num_selected=1')

    accountant = ShuffleAccountant()
    result = accountant.get_deltas(
        params.sigma,
        (params.epsilon,),
        params.num_steps,
        params.num_epochs,
    )[0]
    return float(result)
