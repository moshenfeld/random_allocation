from functools import cache
from dp_accounting import pld, dp_event, rdp
from typing import List

# ==================== PLD ====================
@cache
def poisson_pld(sigma: float,
                num_steps: int,
                num_epochs: int,
                sampling_prob: float,
                discretization: float = 1e-4,
                ) -> pld.privacy_loss_distribution:
    pl_dist = pld.privacy_loss_distribution.from_gaussian_mechanism(standard_deviation=sigma,
                                                                    pessimistic_estimate=True,
                                                                    value_discretization_interval=discretization,
                                                                    sampling_prob=sampling_prob,
                                                                    use_connect_dots=True)
    return pl_dist.self_compose(num_steps*num_epochs)

@cache
def poisson_delta_pld(sigma: float,
                      epsilon: float,
                      num_steps: int,
                      num_selected: int,
                      num_epochs: int,
                      sampling_prob: float = 0.0,
                      discretization: float = 1e-4,
                      ) -> float:
    if sampling_prob == 0.0:
        sampling_prob = num_selected/num_steps
    pld = poisson_pld(sigma=sigma, num_steps=num_steps, num_epochs=num_epochs, sampling_prob=sampling_prob,
                      discretization=discretization)
    return pld.get_delta_for_epsilon(epsilon)

@cache
def poisson_epsilon_pld(sigma: float,
                        delta: float,
                        num_steps: int,
                        num_selected: int,
                        num_epochs: int,
                        sampling_prob: float = 0.0,
                        discretization: float = 1e-4,
                        ) -> float:
    if sampling_prob == 0.0:
        sampling_prob = num_selected/num_steps
    pld = poisson_pld(sigma=sigma, num_steps=num_steps, num_epochs=num_epochs, sampling_prob=sampling_prob,
                      discretization=discretization)
    return pld.get_epsilon_for_delta(delta)

# ==================== RDP ====================
# @cache
def poisson_rdp(sigma: float,
                num_steps: int,
                num_epochs: int,
                sampling_prob: float,
                alpha_orders: List[float],
                ) -> rdp.RdpAccountant:
    accountant = rdp.RdpAccountant(alpha_orders)
    event = dp_event.PoissonSampledDpEvent(sampling_prob, dp_event.GaussianDpEvent(sigma))
    accountant.compose(event, int(num_steps*num_epochs))
    return accountant

# @cache
def poisson_delta_rdp(sigma: float,
                      epsilon: float,
                      num_steps: int,
                      num_selected: int,
                      num_epochs: int,
                      sampling_prob: float = 0.0,
                      alpha_orders: List[float] = None,
                      print_alpha: bool = False,
                      ) -> float:
    if sampling_prob == 0.0:
        sampling_prob = num_selected/num_steps
    accountant = poisson_rdp(sigma=sigma, num_steps=num_steps, num_epochs=num_epochs, sampling_prob=sampling_prob,
                             alpha_orders=alpha_orders)
    delta, used_alpha = accountant.get_delta_and_optimal_order(epsilon)
    if print_alpha:
        print(f'Used alpha: {used_alpha}')
        return delta
    return accountant.get_delta(epsilon)

# @cache
def poisson_epsilon_rdp(sigma: float,
                        delta: float,
                        num_steps: int,
                        num_selected: int,
                        num_epochs: int,
                        sampling_prob: float = 0.0,
                        alpha_orders: List[float] = None,
                        print_alpha: bool = False,
                        ) -> float:
    if sampling_prob == 0.0:
        sampling_prob = num_selected/num_steps
    accountant = poisson_rdp(sigma=sigma, num_steps=num_steps, num_epochs=num_epochs, sampling_prob=sampling_prob,
                             alpha_orders=alpha_orders)
    epsilon, used_alpha = accountant.get_epsilon_and_optimal_order(delta)
    if print_alpha:
        print(f'Used alpha: {used_alpha}')
        return epsilon
    return accountant.get_epsilon(delta)