# Standard library imports
# (none)

# Third-party imports
# (none)

# Local application imports
from random_allocation.random_allocation_scheme.Monte_Carlo_external import *
from random_allocation.comparisons.definitions import PrivacyParams, SchemeConfig

def allocation_delta_lower_bound(params: PrivacyParams, config: SchemeConfig) -> float:
    """
    Compute a lower bound on delta for the allocation scheme.
    
    Args:
        params: Privacy parameters (must include epsilon)
        config: Scheme configuration parameters
    
    Returns:
        Lower bound on delta
    """
    params.validate()
    if params.epsilon is None:
        raise ValueError("Epsilon must be provided to compute delta")
    
    assert(params.num_selected == 1)
    bnb_accountant = BnBAccountant()
    
    return bnb_accountant.get_deltas_lower_bound(
        params.sigma, 
        (params.epsilon), 
        params.num_steps, 
        params.num_epochs
    )[0]

def allocation_epsilon_lower_bound(params: PrivacyParams, config: SchemeConfig) -> float:
    return 0
#     """
#     Compute a lower bound on epsilon for the allocation scheme.
    
#     Args:
#         params: Privacy parameters (must include delta)
#         config: Scheme configuration parameters
    
#     Returns:
#         Lower bound on epsilon
#     """
#     params.validate()
#     if params.delta is None:
#         raise ValueError("Delta must be provided to compute epsilon")
    
#     assert(params.num_selected == 1)
#     #find the epsilon that gives the delta using binary search    
#     optimization_func = lambda eps: allocation_delta_lower_bound(
#         epsilon=eps, 
#         num_steps=num_steps_per_round,
#         Poisson_PLD_obj=Poisson_PLD_obj
#     )
    
#     epsilon = search_function_with_bounds(
#         func=optimization_func, 
#         y_target=params.delta, 
#         bounds=(0, config.epsilon_upper_bound),
#         tolerance=config.epsilon_tolerance, 
#         function_type=FunctionType.DECREASING
#     )