# from functools import cache
import numpy as np
# from computeamplification import numericalanalysis
from .shuffle_external import numericalanalysis
from .local import local_epsilon, bin_search

# @cache
def shuffle_epsilon_analytic(sigma: float,
                             delta: float,
                             num_steps: int,
                             num_selected: int,
                             num_epochs: int,
                             step: float = 100,
                             ) -> float:
    if num_epochs > 1 or num_selected > 1:
        raise ValueError('Shuffle method only supports num_epochs=1 and num_selected=1')
    delta_split = 0.05
    det_eps = local_epsilon(sigma=sigma, delta=delta, num_selected=num_selected, num_epochs=num_epochs)
    local_delta = delta*delta_split/(2*num_steps*(np.exp(2)+1)*(1+np.exp(2)/2))
    local_epsilon_var = local_epsilon(sigma=sigma, delta=local_delta, num_selected=1, num_epochs=1)
    if local_epsilon_var is None or local_epsilon_var > 10:
        return det_eps
    epsilon = numericalanalysis(n=num_steps, epsorig=local_epsilon_var, delta=delta*(1-delta_split), num_iterations=num_epochs,
                                step=step, upperbound=True)
    for _ in range(5):
        local_delta = delta/(2*num_steps*(np.exp(epsilon)+1)*(1+np.exp(local_epsilon_var)/2))
        local_epsilon_var = local_epsilon(sigma, local_delta, num_selected=1, num_epochs=1)
        epsilon = epsilon = numericalanalysis(n=num_steps, epsorig=local_epsilon_var, delta=delta*(1-delta_split),
                                              num_iterations=num_epochs, step=step, upperbound=True)
        delta_bnd = delta*(1-delta_split)+local_delta*num_steps*(np.exp(epsilon)+1)*(1+np.exp(local_epsilon_var)/2)
        if delta_bnd < delta:
            break
    if epsilon > det_eps:
        return det_eps
    return epsilon

# @cache
def shuffle_delta_analytic(sigma: float,
                           epsilon: float,
                           num_steps: int,
                           num_selected: int,
                           num_epochs: int,
                           step: float = 100,
                           ) -> float:
    if num_epochs > 1 or num_selected > 1:
        raise ValueError('Shuffle method only supports num_epochs=1 and num_selected=1')
    return bin_search(lambda delta: shuffle_epsilon_analytic(sigma=sigma, delta=delta, num_steps=num_steps, num_selected=num_selected,
                                                             num_epochs=num_epochs, step=step),
                      0, 1, epsilon, increasing=False)