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