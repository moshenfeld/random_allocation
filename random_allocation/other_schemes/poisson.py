# from functools import cache
from dp_accounting import pld, dp_event, rdp

from typing import List

# ==================== PLD ====================
# @cache
def poisson_pld(sigma: float,
                num_steps: int,
                num_epochs: int,
                sampling_prob: float,
                discretization: float,
                direction: str,
                ) -> pld.privacy_loss_distribution:
    """
    Calculate the privacy loss distribution (PLD) for the Poisson scheme with the Gaussian mechanism.

    Parameters:
    - sigma: The standard deviation of the Gaussian mechanism.
    - num_steps: The number of steps in each epoch.
    - sampling_prob: The probability of sampling.
    - num_epochs: The number of epochs.
    - discretization: The discretization interval for the PLD.
    - direction: The direction of the PLD. Can be 'add', 'remove', or 'both'.
    """
    gauss_pld = pld.privacy_loss_distribution.from_gaussian_mechanism(standard_deviation=sigma,
                                                                      value_discretization_interval=discretization,
                                                                      pessimistic_estimate=True,
                                                                      sampling_prob=sampling_prob,
                                                                      use_connect_dots=True)
    zero_delta_pmf = pld.privacy_loss_distribution.pld_pmf.create_pmf(loss_probs={-10: 1.0},
                                                                      discretization=discretization,
                                                                      infinity_mass=0,
                                                                      pessimistic_estimate=True)
    if direction == "add":
        pld_single = pld.privacy_loss_distribution.PrivacyLossDistribution(zero_delta_pmf, gauss_pld._pmf_add)
    elif direction == "remove":
        pld_single = pld.privacy_loss_distribution.PrivacyLossDistribution(gauss_pld._pmf_remove, zero_delta_pmf)
    elif direction == "both":
        pld_single = gauss_pld
    return pld_single.self_compose(num_steps*num_epochs)

# @cache
def poisson_delta_pld(sigma: float,
                      epsilon: float,
                      num_steps: int,
                      num_selected: int,
                      num_epochs: int,
                      sampling_prob: float = 0.0,
                      discretization: float = 1e-4,
                      direction: str = 'both',
                      ) -> float:
    """
    Calculate the delta value for the Poisson scheme with the Gaussian mechanism based on PLD.

    Parameters:
    - sigma: The standard deviation of the Gaussian mechanism.
    - epsilon: The privacy parameter.
    - num_steps: The number of steps in each epoch.
    - num_selected: The number of selected items.
    - num_epochs: The number of epochs.
    - sampling_prob: The probability of sampling.
    - discretization: The discretization interval for the PLD.
    """
    if sampling_prob == 0.0:
        sampling_prob = num_selected/num_steps
    pld = poisson_pld(sigma=sigma, num_steps=num_steps, num_epochs=num_epochs, sampling_prob=sampling_prob,
                      discretization=discretization, direction=direction)
    return pld.get_delta_for_epsilon(epsilon)

# @cache
def poisson_epsilon_pld(sigma: float,
                        delta: float,
                        num_steps: int,
                        num_selected: int,
                        num_epochs: int,
                        sampling_prob: float = 0.0,
                        discretization: float = 1e-4,
                        direction: str = 'both',
                        ) -> float:
    """
    Calculate the epsilon value for the Poisson scheme with the Gaussian mechanism based on PLD.

    Parameters:
    - sigma: The standard deviation of the Gaussian mechanism.
    - delta: The privacy profile bound.
    - num_steps: The number of steps in each epoch.
    - num_selected: The number of selected items.
    - num_epochs: The number of epochs.
    - sampling_prob: The probability of sampling.
    - discretization: The discretization interval for the PLD.
    """
    if sampling_prob == 0.0:
        sampling_prob = num_selected/num_steps
    pld = poisson_pld(sigma=sigma, num_steps=num_steps, num_epochs=num_epochs, sampling_prob=sampling_prob,
                      discretization=discretization, direction=direction)
    return pld.get_epsilon_for_delta(delta)

# ==================== RDP ====================
# @cache
def poisson_rdp(sigma: float,
                num_steps: int,
                num_epochs: int,
                sampling_prob: float,
                alpha_orders: List[float],
                ) -> rdp.RdpAccountant:
    """
    Create an RDP accountant for the Poisson scheme with the Gaussian mechanism.

    Parameters:
    - sigma: The standard deviation of the Gaussian mechanism.
    - num_steps: The number of steps in each epoch.
    - num_epochs: The number of epochs.
    - sampling_prob: The probability of sampling.
    - alpha_orders: The list of alpha orders for RDP.
    """
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
    """
    Calculate the delta value for the Poisson scheme with the Gaussian mechanism based on RDP.

    Parameters:
    - sigma: The standard deviation of the Gaussian mechanism.
    - epsilon: The privacy parameter.
    - num_steps: The number of steps in each epoch.
    - num_selected: The number of steps that an element is used per epoch
    - num_epochs: The number of epochs.
    - sampling_prob: The probability of sampling.
    - alpha_orders: The list of alpha orders for RDP.
    - print_alpha: Whether to print the used alpha order.
    """
    # Default sampling probability is num_selected/num_steps
    if sampling_prob == 0.0:
        sampling_prob = num_selected/num_steps
    accountant = poisson_rdp(sigma=sigma, num_steps=num_steps, num_epochs=num_epochs, sampling_prob=sampling_prob,
                             alpha_orders=alpha_orders)
    if print_alpha:
        delta, used_alpha = accountant.get_delta_and_optimal_order(epsilon)
        print(f'sigma: {sigma}, num_steps: {num_steps}, num_epochs: {num_epochs}, sampling_prob: {sampling_prob}, used_alpha: {used_alpha}')
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
    """
    Calculate the epsilon value for the Poisson scheme with the Gaussian mechanism based on RDP.

    Parameters:
    - sigma: The standard deviation of the Gaussian mechanism.
    - delta: The privacy profile bound.
    - num_steps: The number of steps in each epoch.
    - num_selected: The number of steps that an element is used per epoch
    - num_epochs: The number of epochs.
    - sampling_prob: The probability of sampling.
    - alpha_orders: The list of alpha orders for RDP.
    - print_alpha: Whether to print the used alpha order.
    """
    # Default sampling probability is num_selected/num_steps
    if sampling_prob == 0.0:
        sampling_prob = num_selected/num_steps
    accountant = poisson_rdp(sigma=sigma, num_steps=num_steps, num_epochs=num_epochs, sampling_prob=sampling_prob,
                             alpha_orders=alpha_orders)
    if print_alpha:
        epsilon, used_alpha = accountant.get_epsilon_and_optimal_order(delta)
        print(f'sigma: {sigma}, num_steps: {num_steps}, num_epochs: {num_epochs}, sampling_prob: {sampling_prob}, used_alpha: {used_alpha}')
        return epsilon
    return accountant.get_epsilon(delta)