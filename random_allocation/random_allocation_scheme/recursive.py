# Standard library imports
from typing import Callable, Dict, Any, List, Tuple, cast

# Third-party imports
import numpy as np
from dp_accounting.pld.privacy_loss_distribution import PrivacyLossDistribution

# Local application imports
from random_allocation.comparisons.definitions import PrivacyParams, SchemeConfig, Direction
from random_allocation.comparisons.utils import search_function_with_bounds, FunctionType
from random_allocation.other_schemes.poisson import Poisson_PLD, Poisson_delta_PLD
from random_allocation.random_allocation_scheme.decomposition import allocation_delta_decomposition
from random_allocation.random_allocation_scheme.direct import allocation_delta_direct
# Type aliases
NumericFunction = Callable[[float], float]

def allocation_epsilon_recursive_inner(sigma, delta, num_steps, num_epochs, discretization, optimization_func):
    epsilon = np.inf

    # Find a gamma such that the Poisson term with sampling probability e^(2*gamma)/num_steps_per_round
    # is less than delta/2
    gamma_result = search_function_with_bounds(
        func=optimization_func, 
        y_target=delta/2, 
        bounds=(1e-5, 10),
        tolerance=1e-2, 
        function_type=FunctionType.DECREASING
    )
    
    # If we find a gamma, we can compute the sampling probability
    if gamma_result is not None:
        sampling_prob = np.exp(2 * gamma_result)/num_steps

        # if the induced sampling probability is small enough, we can compute the corresponding Poisson epsilon
        if sampling_prob <= np.sqrt(1/num_steps):
            Poisson_PLD_final = Poisson_PLD(
                sigma=sigma,
                num_steps=num_steps,
                num_epochs=num_epochs,
                sampling_prob=sampling_prob,
                discretization=discretization,
                direction="remove"
            )
            epsilon = float(Poisson_PLD_final.get_epsilon_for_delta(delta/2))
    return epsilon

def allocation_epsilon_recursive(params: PrivacyParams,
                                config: SchemeConfig = SchemeConfig(),
                                direction: Direction = Direction.BOTH,
                                ) -> float:
    """
    Compute epsilon for the recursive allocation scheme.
    
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
        
    num_steps_per_round = int(np.ceil(params.num_steps/params.num_selected))
    num_rounds = int(np.ceil(params.num_steps/num_steps_per_round))
    lambda_val = 1 - (1-1.0/num_steps_per_round)**num_steps_per_round
    
    Poisson_PLD_base = Poisson_PLD(
        sigma=params.sigma, 
        num_steps=num_steps_per_round, 
        num_epochs=num_rounds*params.num_epochs, 
        sampling_prob=1.0/num_steps_per_round, 
        discretization=config.discretization, 
        direction="add"
    )
    
    if direction != Direction.ADD:
        optimization_func = lambda eps: Poisson_PLD_base.get_delta_for_epsilon(-np.log(1-lambda_val*(1-np.exp(-eps))))\
                                        *(1/(lambda_val*(np.exp(eps) -1)) - np.exp(-eps))
        epsilon_remove = allocation_epsilon_recursive_inner(params.sigma, params.delta, num_steps_per_round,
                                                            num_rounds*params.num_epochs, config.discretization, optimization_func)
    
    if direction != Direction.REMOVE:
        optimization_func = lambda eps: Poisson_PLD_base.get_delta_for_epsilon(-np.log(1-lambda_val*(1-np.exp(-eps))))\
                                        *(1/(lambda_val*(np.exp(eps) -1)) - np.exp(-eps))
        epsilon_add = allocation_epsilon_recursive_inner(params.sigma, params.delta, num_steps_per_round,
                                                         num_rounds*params.num_epochs, config.discretization, optimization_func)
    
    if direction == Direction.ADD:
        assert 'epsilon_add' in locals(), "Failed to compute epsilon_add"
        return epsilon_add
    if direction == Direction.REMOVE:
        assert 'epsilon_remove' in locals(), "Failed to compute epsilon_remove"
        return epsilon_remove
    
    # Both directions, return max of the two
    assert 'epsilon_add' in locals() and 'epsilon_remove' in locals(), "Failed to compute either epsilon_add or epsilon_remove"
    return max(epsilon_add, epsilon_remove)

def allocation_delta_recursive(params: PrivacyParams,
                               config: SchemeConfig = SchemeConfig(),
                               direction: Direction = Direction.BOTH,
                               ) -> float:
    """
    Compute delta for the recursive allocation scheme.
    
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
    
    delta_add: float  # type annotation without initialization
    delta_remove: float  # type annotation without initialization
        
    if direction != Direction.ADD:
        gamma = min(params.epsilon/2, np.log(params.num_steps)/4)
        eta = np.exp(2*gamma)/params.num_steps
        params2 = PrivacyParams(
            sigma=params.sigma,
            num_steps=params.num_steps,
            num_selected=params.num_selected,
            num_epochs=params.num_epochs,
            epsilon=gamma,
            delta=None
        )
        delta_add_decomposition = allocation_delta_decomposition(params=params2, config=config, direction=Direction.ADD)
        delta_add_direct = allocation_delta_direct(params=params2, config=config, direction=Direction.ADD)
        delta_remove = Poisson_delta_PLD(
            params=params, 
            config=config, 
            sampling_prob=eta, 
            direction=Direction.REMOVE
        ) + 1/(np.exp(2*gamma)-np.exp(gamma)) * min(delta_add_decomposition, delta_add_direct)
    
    if direction != Direction.REMOVE:
        if params.epsilon > 0.5:
            delta_add = allocation_delta_decomposition(params=params, config=config, direction=Direction.ADD)
        else:
            gamma = min(params.epsilon*2, np.log(params.num_steps)/4)
            eta = np.exp(2*gamma)/params.num_steps
            params2 = PrivacyParams(
                sigma=params.sigma,
                num_steps=params.num_steps,
                num_selected=params.num_selected,
                num_epochs=params.num_epochs,
                epsilon=gamma,
                delta=None
            )
            delta_add_decomposition = allocation_delta_decomposition(params=params2, config=config, direction=Direction.ADD)
            delta_add_direct = allocation_delta_direct(params=params2, config=config, direction=Direction.ADD)
            delta_add = Poisson_delta_PLD(
                params=params, 
                config=config, 
                sampling_prob=eta, 
                direction=Direction.ADD
            ) + np.exp(gamma)/(np.exp(gamma)-1) * min(delta_add_decomposition, delta_add_direct)

    if direction == Direction.ADD:
        assert 'delta_add' in locals(), "Failed to compute delta_add"
        return float(delta_add)
    if direction == Direction.REMOVE:
        assert 'delta_remove' in locals(), "Failed to compute delta_remove"
        return float(delta_remove)
    # Both directions, return max of the two
    assert 'delta_add' in locals() and 'delta_remove' in locals(), "Failed to compute either delta_add or delta_remove"
    return float(max(delta_add, delta_remove))
