from random_allocation.random_allocation_scheme.Monte_Carlo_external import *

def allocation_delta_Monte_Carlo(sigma: float,
                                 epsilon: float,
                                 num_steps: int,
                                 num_selected: int,
                                 num_epochs: int,
                                 direction: str = 'both',
                                 use_order_stats: bool = True,
                                 use_mean: bool = False,
                                 ) -> float:
    assert(num_selected == 1)
    bnb_accountant = BnBAccountant()
    error_prob = 0.01
    if direction != 'add':
        adjacency_type = AdjacencyType.REMOVE
        if use_order_stats:
            sample_size = 500_000
            order_stats_encoding = (1, 100, 1, 100, 500, 10, 500, 1000, 50)
            order_stats_seq = get_order_stats_seq_from_encoding(order_stats_encoding, num_steps)
            delta_estimate = bnb_accountant.estimate_order_stats_deltas(sigma, [epsilon], num_steps, sample_size, order_stats_seq,
                                                                        num_epochs, adjacency_type)[0]
        else:
            sample_size = 100_000
            delta_estimate = bnb_accountant.estimate_deltas(sigma, [epsilon], num_steps, sample_size, num_epochs, adjacency_type, 
                                                            use_importance_sampling=True)[0]
        delta_remove = delta_estimate.mean if use_mean else delta_estimate.get_upper_confidence_bound(error_prob)
    if direction != 'remove':
        adjacency_type = AdjacencyType.ADD
        if use_order_stats:
            sample_size = 500_000
            order_stats_encoding = (1, 100, 1, 100, 500, 10, 500, 1000, 50)
            order_stats_seq = get_order_stats_seq_from_encoding(order_stats_encoding, num_steps)
            delta_estimate = bnb_accountant.estimate_order_stats_deltas(sigma, [epsilon], num_steps, sample_size, order_stats_seq,
                                                                        num_epochs, adjacency_type)[0]
        else:
            sample_size = 100_000
            delta_estimate = bnb_accountant.estimate_deltas(sigma, [epsilon], num_steps, sample_size, num_epochs, adjacency_type, 
                                                            use_importance_sampling=True)[0]
        delta_add = delta_estimate.mean if use_mean else delta_estimate.get_upper_confidence_bound(error_prob)
    if direction == 'add':
        return delta_add
    if direction == 'remove':
        return delta_remove
    return max(delta_add, delta_remove)

def allocation_delta_lower(sigma: float,
                           epsilon: float,
                           num_steps: int,
                           num_selected: int,
                           num_epochs: int,
                           ) -> float:
    assert(num_selected == 1)
    bnb_accountant = BnBAccountant()
    return bnb_accountant.get_deltas_lower_bound(sigma, (epsilon), num_steps, num_epochs)[0]
