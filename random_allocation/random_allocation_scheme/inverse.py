import math

from random_allocation.other_schemes.local import Gaussian_epsilon, Gaussian_delta

def allocation_epsilon_inverse(sigma: float,
                               delta: float,
                               num_steps: int,
                               ) -> float:
    return Gaussian_epsilon(sigma=sigma*math.sqrt(num_steps), delta=delta) + (1-1.0/num_steps)/(2*sigma**2)

def allocation_delta_inverse(sigma: float,
                             epsilon: float,
                             num_steps: int,
                             ) -> float:
    return Gaussian_delta(sigma=sigma*math.sqrt(num_steps), epsilon=epsilon - (1-1.0/num_steps)/(2*sigma**2))