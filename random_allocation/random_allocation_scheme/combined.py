from random_allocation.other_schemes.local import local_epsilon, bin_search
from random_allocation.other_schemes.poisson import poisson_epsilon_pld
from random_allocation.random_allocation_scheme.inverse import allocation_epsilon_inverse, allocation_delta_inverse
from random_allocation.random_allocation_scheme.analytic import allocation_epsilon_analytic
from random_allocation.random_allocation_scheme.decomposition import allocation_epsilon_decomposition
from random_allocation.random_allocation_scheme.RDP import allocation_epsilon_rdp



def allocation_delta_combined(sigma: float,
                              epsilon: float,
                              num_steps: int,
                              num_selected: int,
                              num_epochs: int,
                              discretization: float = 1e-4,
                              ) -> float:
    return 0

def allocation_epsilon_combined(sigma: float,
                                delta: float,
                                num_steps: int,
                                num_selected: int,
                                num_epochs: int,
                                direction: str = 'both',
                                discretization: float = 1e-4,
                                epsilon_tolerance: float = 1e-3,
                                epsilon_upper_bound: float = 100,
                                min_alpha: int = 2,
                                max_alpha: int = 50,
                                print_alpha: bool = False,
                                ) -> float:
    if direction != 'add':
        epsilon_remove_analytic = allocation_epsilon_analytic(sigma=sigma, delta=delta, num_steps=num_steps,
                                                              num_selected=num_selected, num_epochs=num_epochs, direction='remove', discretization=discretization)
        epsilon_remove_decompose = allocation_epsilon_decomposition(sigma=sigma, delta=delta, num_steps=num_steps,
                                                                    num_selected=num_selected, num_epochs=num_epochs,
                                                                    direction='remove', discretization=discretization, epsilon_tolerance=epsilon_tolerance, epsilon_upper_bound=epsilon_upper_bound)
        epsilon_remove_RDP = allocation_epsilon_rdp(sigma=sigma, delta=delta, num_steps=num_steps, 
                                                    num_selected=num_selected, num_epochs=num_epochs, direction='remove',min_alpha=min_alpha, max_alpha=max_alpha, print_alpha=print_alpha)
        epsilon_remove = min(epsilon_remove_analytic, epsilon_remove_decompose, epsilon_remove_RDP)
    if direction != 'remove':
        epsilon_add_analytic = allocation_epsilon_analytic(sigma=sigma, delta=delta, num_steps=num_steps,
                                                           num_selected=num_selected, num_epochs=num_epochs, direction='add', discretization=discretization)
        epsilon_add_decompose = allocation_epsilon_decomposition(sigma=sigma, delta=delta, num_steps=num_steps,
                                                                 num_selected=num_selected, num_epochs=num_epochs,
                                                                 direction='add', discretization=discretization, epsilon_tolerance=epsilon_tolerance, epsilon_upper_bound=epsilon_upper_bound)
        epsilon_add_RDP = allocation_epsilon_rdp(sigma=sigma, delta=delta, num_steps=num_steps, 
                                                 num_selected=num_selected, num_epochs=num_epochs, direction='dd',min_alpha=min_alpha, max_alpha=max_alpha, print_alpha=print_alpha)
        epsilon_add_inverse = allocation_epsilon_inverse(sigma=sigma, delta=delta, num_steps=num_steps)
        epsilon_add = min(epsilon_add_analytic, epsilon_add_decompose, epsilon_add_RDP, epsilon_add_inverse)
    if direction == 'add':
        return epsilon_add
    if direction == 'remove':
        return epsilon_remove
    return max(epsilon_remove, epsilon_add) 