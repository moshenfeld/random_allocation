from typing import List

from dp_accounting import pld, dp_event, rdp
from random_allocation.comparisons.definitions import PrivacyParams, SchemeConfig


# ==================== PLD ====================
def Poisson_PLD(sigma: float,
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
    - discretization: The discretization interval for the pld.
    - direction: The direction of the pld. Can be 'add', 'remove', or 'both'.
    """
    Gauss_PLD = pld.privacy_loss_distribution.from_gaussian_mechanism(standard_deviation=sigma,
                                                                      value_discretization_interval=discretization,
                                                                      pessimistic_estimate=True,
                                                                      sampling_prob=sampling_prob,
                                                                      use_connect_dots=True)
    zero_delta_pmf = pld.privacy_loss_distribution.pld_pmf.create_pmf(loss_probs={-10: 1.0},
                                                                      discretization=discretization,
                                                                      infinity_mass=0,
                                                                      pessimistic_estimate=True)
    if direction == "add":
        PLD_single = pld.privacy_loss_distribution.PrivacyLossDistribution(zero_delta_pmf, Gauss_PLD._pmf_add)
    elif direction == "remove":
        PLD_single = pld.privacy_loss_distribution.PrivacyLossDistribution(Gauss_PLD._pmf_remove, zero_delta_pmf)
    elif direction == "both":
        PLD_single = Gauss_PLD
    return PLD_single.self_compose(num_steps*num_epochs)

def Poisson_delta_PLD(params: PrivacyParams,
                      config: SchemeConfig = SchemeConfig(),
                      sampling_prob: float = 0.0,
                      ) -> float:
    """
    Calculate the delta value for the Poisson scheme with the Gaussian mechanism based on pld.

    Parameters:
    - params: Privacy parameters
    - config: Scheme configuration
    """
    params.validate()
    if params.epsilon is None:
        raise ValueError("Epsilon must be provided to compute delta")
    
    if sampling_prob == 0.0:
        sampling_prob = params.num_selected / params.num_steps
    
    PLD = Poisson_PLD(
        sigma=params.sigma, 
        num_steps=params.num_steps, 
        num_epochs=params.num_epochs, 
        sampling_prob=sampling_prob,
        discretization=config.discretization, 
        direction=config.direction
    )
    return PLD.get_delta_for_epsilon(params.epsilon)

def Poisson_epsilon_PLD(params: PrivacyParams,
                        config: SchemeConfig = SchemeConfig(),
                        sampling_prob: float = 0.0,
                        ) -> float:
    """
    Calculate the epsilon value for the Poisson scheme with the Gaussian mechanism based on pld.

    Parameters:
    - params: Privacy parameters
    - config: Scheme configuration
    """
    params.validate()
    if params.delta is None:
        raise ValueError("Delta must be provided to compute epsilon")
    
    if sampling_prob == 0.0:
        sampling_prob = params.num_selected / params.num_steps
    
    PLD = Poisson_PLD(
        sigma=params.sigma, 
        num_steps=params.num_steps, 
        num_epochs=params.num_epochs, 
        sampling_prob=sampling_prob,
        discretization=config.discretization, 
        direction=config.direction
    )
    return PLD.get_epsilon_for_delta(params.delta)

# ==================== RDP ====================
def Poisson_RDP(sigma: float,
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
    - alpha_orders: The list of alpha orders for rdp.
    """
    accountant = rdp.RdpAccountant(alpha_orders)
    event = dp_event.PoissonSampledDpEvent(sampling_prob, dp_event.GaussianDpEvent(sigma))
    accountant.compose(event, int(num_steps*num_epochs))
    return accountant

def Poisson_delta_RDP(params: PrivacyParams,
                      config: SchemeConfig = SchemeConfig(),
                      ) -> float:
    """
    Calculate the delta value for the Poisson scheme with the Gaussian mechanism based on rdp.

    Parameters:
    - params: Privacy parameters
    - config: Scheme configuration
    """
    params.validate()
    if params.epsilon is None:
        raise ValueError("Epsilon must be provided to compute delta")
    
    sampling_prob = params.num_selected / params.num_steps
    alpha_orders = list(range(config.min_alpha, config.max_alpha + 1))
    
    accountant = Poisson_RDP(
        sigma=params.sigma, 
        num_steps=params.num_steps, 
        num_epochs=params.num_epochs, 
        sampling_prob=sampling_prob,
        alpha_orders=alpha_orders
    )
    
    if config.print_alpha:
        delta, used_alpha = accountant.get_delta_and_optimal_order(params.epsilon)
        print(f'sigma: {params.sigma}, num_steps: {params.num_steps}, num_epochs: {params.num_epochs}, '
              f'sampling_prob: {sampling_prob}, used_alpha: {used_alpha}')
        return delta
    
    return accountant.get_delta(params.epsilon)

def Poisson_epsilon_RDP(params: PrivacyParams,
                        config: SchemeConfig = SchemeConfig(),
                        ) -> float:
    """
    Calculate the epsilon value for the Poisson scheme with the Gaussian mechanism based on rdp.

    Parameters:
    - params: Privacy parameters
    - config: Scheme configuration
    """
    params.validate()
    if params.delta is None:
        raise ValueError("Delta must be provided to compute epsilon")
    
    sampling_prob = params.num_selected / params.num_steps
    alpha_orders = list(range(config.min_alpha, config.max_alpha + 1))
    
    accountant = Poisson_RDP(
        sigma=params.sigma, 
        num_steps=params.num_steps, 
        num_epochs=params.num_epochs, 
        sampling_prob=sampling_prob,
        alpha_orders=alpha_orders
    )
    
    if config.print_alpha:
        epsilon, used_alpha = accountant.get_epsilon_and_optimal_order(params.delta)
        print(f'sigma: {params.sigma}, num_steps: {params.num_steps}, num_epochs: {params.num_epochs}, '
              f'sampling_prob: {sampling_prob}, used_alpha: {used_alpha}')
        return epsilon
    
    return accountant.get_epsilon(params.delta)

# For backward compatibility
def _Poisson_delta_PLD_legacy(sigma: float,
                      epsilon: float,
                      num_steps: int,
                      num_selected: int,
                      num_epochs: int,
                      sampling_prob: float = 0.0,
                      discretization: float = 1e-4,
                      direction: str = 'both',
                      ) -> float:
    """Legacy function for backward compatibility"""
    temp_params = PrivacyParams(
        sigma=sigma,
        epsilon=epsilon,
        delta=None,
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs
    )
    
    temp_config = SchemeConfig(
        discretization=discretization,
        direction=direction
    )
    
    return Poisson_delta_PLD(params=temp_params, config=temp_config)

def _Poisson_epsilon_PLD_legacy(sigma: float,
                        delta: float,
                        num_steps: int,
                        num_selected: int,
                        num_epochs: int,
                        sampling_prob: float = 0.0,
                        discretization: float = 1e-4,
                        direction: str = 'both',
                        ) -> float:
    """Legacy function for backward compatibility"""
    temp_params = PrivacyParams(
        sigma=sigma,
        epsilon=None,
        delta=delta,
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs
    )
    
    temp_config = SchemeConfig(
        discretization=discretization,
        direction=direction
    )
    
    return Poisson_epsilon_PLD(params=temp_params, config=temp_config)

def _Poisson_delta_RDP_legacy(sigma: float,
                      epsilon: float,
                      num_steps: int,
                      num_selected: int,
                      num_epochs: int,
                      sampling_prob: float = 0.0,
                      alpha_orders: List[float] = None,
                      print_alpha: bool = False,
                      ) -> float:
    """Legacy function for backward compatibility"""
    temp_params = PrivacyParams(
        sigma=sigma,
        epsilon=epsilon,
        delta=None,
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs
    )
    
    temp_config = SchemeConfig(
        min_alpha=alpha_orders[0] if alpha_orders else 2,
        max_alpha=alpha_orders[-1] if alpha_orders else 50,
        print_alpha=print_alpha
    )
    
    return Poisson_delta_RDP(params=temp_params, config=temp_config)

def _Poisson_epsilon_RDP_legacy(sigma: float,
                        delta: float,
                        num_steps: int,
                        num_selected: int,
                        num_epochs: int,
                        sampling_prob: float = 0.0,
                        alpha_orders: List[float] = None,
                        print_alpha: bool = False,
                        ) -> float:
    """Legacy function for backward compatibility"""
    temp_params = PrivacyParams(
        sigma=sigma,
        epsilon=None,
        delta=delta,
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs
    )
    
    temp_config = SchemeConfig(
        min_alpha=alpha_orders[0] if alpha_orders else 2,
        max_alpha=alpha_orders[-1] if alpha_orders else 50,
        print_alpha=print_alpha
    )
    
    return Poisson_epsilon_RDP(params=temp_params, config=temp_config)