from random_allocation.random_allocation_scheme.Monte_Carlo_external import *
from random_allocation.comparisons.definitions import PrivacyParams, SchemeConfig

def allocation_delta_Monte_Carlo(params: PrivacyParams,
                                config: SchemeConfig = SchemeConfig(),
                                use_order_stats: bool = True,
                                use_mean: bool = False,
                                ) -> float:
    """
    Compute delta using Monte Carlo simulation for the allocation scheme.
    
    Args:
        params: Privacy parameters (must include epsilon)
        config: Scheme configuration parameters
        use_order_stats: Whether to use order statistics
        use_mean: Whether to use mean or upper confidence bound
    
    Returns:
        Computed delta value
    """
    params.validate()
    if params.epsilon is None:
        raise ValueError("Epsilon must be provided to compute delta")
    
    assert(params.num_selected == 1)
    bnb_accountant = BnBAccountant()
    error_prob = 0.01
    
    if config.direction != 'add':
        adjacency_type = AdjacencyType.REMOVE
        if use_order_stats:
            sample_size = 500_000
            order_stats_encoding = (1, 100, 1, 100, 500, 10, 500, 1000, 50)
            order_stats_seq = get_order_stats_seq_from_encoding(order_stats_encoding, params.num_steps)
            delta_estimate = bnb_accountant.estimate_order_stats_deltas(
                params.sigma, 
                [params.epsilon], 
                params.num_steps, 
                sample_size, 
                order_stats_seq,
                params.num_epochs, 
                adjacency_type
            )[0]
        else:
            sample_size = 100_000
            delta_estimate = bnb_accountant.estimate_deltas(
                params.sigma, 
                [params.epsilon], 
                params.num_steps, 
                sample_size, 
                params.num_epochs, 
                adjacency_type, 
                use_importance_sampling=True
            )[0]
        delta_remove = delta_estimate.mean if use_mean else delta_estimate.get_upper_confidence_bound(error_prob)
    
    if config.direction != 'remove':
        adjacency_type = AdjacencyType.ADD
        if use_order_stats:
            sample_size = 500_000
            order_stats_encoding = (1, 100, 1, 100, 500, 10, 500, 1000, 50)
            order_stats_seq = get_order_stats_seq_from_encoding(order_stats_encoding, params.num_steps)
            delta_estimate = bnb_accountant.estimate_order_stats_deltas(
                params.sigma, 
                [params.epsilon], 
                params.num_steps, 
                sample_size, 
                order_stats_seq,
                params.num_epochs, 
                adjacency_type
            )[0]
        else:
            sample_size = 100_000
            delta_estimate = bnb_accountant.estimate_deltas(
                params.sigma, 
                [params.epsilon], 
                params.num_steps, 
                sample_size, 
                params.num_epochs, 
                adjacency_type, 
                use_importance_sampling=True
            )[0]
        delta_add = delta_estimate.mean if use_mean else delta_estimate.get_upper_confidence_bound(error_prob)
    
    if config.direction == 'add':
        return delta_add
    if config.direction == 'remove':
        return delta_remove
    return max(delta_add, delta_remove)

def allocation_delta_lower(params: PrivacyParams) -> float:
    """
    Compute a lower bound on delta for the allocation scheme.
    
    Args:
        params: Privacy parameters (must include epsilon)
    
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
