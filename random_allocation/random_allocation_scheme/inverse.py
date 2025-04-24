import math

from random_allocation.other_schemes.local import Gaussian_epsilon, Gaussian_delta

def allocation_epsilon_inverse(sigma: float,
                               delta: float,
                               num_steps: int,
                               num_selected: int,
                               num_epochs: int,
                               ) -> float:
    num_steps_per_round = int(math.ceil(num_steps/num_selected))
    num_rounds = int(math.ceil(num_steps/num_steps_per_round))
    return Gaussian_epsilon(sigma=sigma*math.sqrt(num_steps_per_round/(num_epochs*num_rounds)), delta=delta) + (1-1.0/num_steps_per_round)/(2*sigma**2)

def allocation_delta_inverse(sigma: float,
                             epsilon: float,
                             num_steps: int,
                             num_selected: int,
                             num_epochs: int,
                             ) -> float:
    num_steps_per_round = int(math.ceil(num_steps/num_selected))
    num_rounds = int(math.ceil(num_steps/num_steps_per_round))
    return Gaussian_delta(sigma=sigma*math.math.sqrt(num_steps_per_round/(num_epochs*num_rounds)), epsilon=epsilon - (1-1.0/num_steps_per_round)/(2*sigma**2))