from random_allocation.random_allocation_scheme.Monte_Carlo_external import *
from random_allocation.comparisons.definitions import PrivacyParams, SchemeConfig

def Monte_Carlo_estimation(params: PrivacyParams,
                           config: SchemeConfig,
                           adjacency_type: AdjacencyType,
                           ) -> float:
    bnb_accountant = BnBAccountant()
    if config.MC_use_order_stats:
        order_stats_encoding = (1, 100, 1, 100, 500, 10, 500, 1000, 50)
        order_stats_seq = get_order_stats_seq_from_encoding(order_stats_encoding, params.num_steps)
        delta_estimate = bnb_accountant.estimate_order_stats_deltas(
            params.sigma, 
            [params.epsilon], 
            params.num_steps, 
            config.MC_sample_size, 
            order_stats_seq,
            params.num_epochs, 
            adjacency_type
        )[0]
    else:
        delta_estimate = bnb_accountant.estimate_deltas(
            params.sigma, 
            [params.epsilon], 
            params.num_steps, 
            config.MC_sample_size, 
            params.num_epochs, 
            adjacency_type, 
            use_importance_sampling=True
        )[0]
    if config.MC_use_mean:
        return delta_estimate.mean
    else:
        return delta_estimate.get_upper_confidence_bound(1-config.MC_conf_level)

def allocation_delta_MC(params: PrivacyParams, config: SchemeConfig) -> float:
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

    if config.direction != 'add':
        delta_remove = Monte_Carlo_estimation(params, config, AdjacencyType.REMOVE)
    if config.direction != 'remove':
        delta_add    = Monte_Carlo_estimation(params, config, AdjacencyType.ADD)
    
    if config.direction == 'add':
        return delta_add
    if config.direction == 'remove':
        return delta_remove
    return max(delta_add, delta_remove)
