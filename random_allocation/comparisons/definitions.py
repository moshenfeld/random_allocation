from dataclasses import dataclass
from typing import Callable, Dict, Any, List
import numpy as np
import pandas as pd

from random_allocation.other_schemes.local import local_epsilon, local_delta
from random_allocation.other_schemes.poisson import poisson_epsilon_pld, poisson_delta_pld, poisson_epsilon_rdp, poisson_delta_rdp
from random_allocation.other_schemes.shuffle import shuffle_epsilon_analytic, shuffle_delta_analytic
from random_allocation.random_allocation_scheme import allocation_epsilon_analytic, allocation_delta_analytic
from random_allocation.random_allocation_scheme import allocation_epsilon_rdp, allocation_delta_rdp
from random_allocation.random_allocation_scheme import allocation_epsilon_rdp_loose, allocation_delta_rdp_loose
from random_allocation.random_allocation_scheme import allocation_epsilon_decomposition, allocation_delta_decomposition

#======================= Variables =======================
EPSILON = 'epsilon'
DELTA = 'delta'
SIGMA = 'sigma'
NUM_STEPS = 'num_steps'
NUM_SELECTED = 'num_selected'
NUM_EPOCHS = 'num_epochs'
VARIABLES = [EPSILON, DELTA, SIGMA, NUM_STEPS, NUM_SELECTED, NUM_EPOCHS]

names_dict = {EPSILON: '$\\varepsilon$', DELTA: '$\\delta$', SIGMA: '$\\sigma$', NUM_STEPS: '$t$', NUM_SELECTED: '$k$',
              NUM_EPOCHS: '$E$'}

#===================== Configuration =====================
NUM_EXP = 'num_experiments'
DISCRETIZATION = 'discretization'
MIN_ALPHA = 'min_alpha'
MAX_ALPHA = 'max_alpha'
CONFIGS = [NUM_EXP, DISCRETIZATION, MIN_ALPHA, MAX_ALPHA]
ALPHA_ORDERS = 'alpha_orders'

# ======================= Schemes =======================
LOCAL = 'Local'
POISSON = 'Poisson'
ALLOCATION = 'allocation'
SHUFFLE = 'Shuffle'

colors_dict = {LOCAL: '#FF0000', POISSON: '#2BB22C', ALLOCATION: '#157DED', SHUFFLE: '#FF00FF'}

# ======================= Computation =======================
ANALYTIC = 'Analytic'
MONTE_CARLO = 'Monte Carlo'
PLD = 'PLD'
RDP = 'RDP'
DECOMPOSITION = 'Decomposition'

# ======================= Methods =======================
POISSON_PLD                 = f'{POISSON} ({PLD})'
POISSON_RDP                 = f'{POISSON} ({RDP})'
ALLOCATION_ANALYTIC         = f'{ALLOCATION} (Our - {ANALYTIC})'
ALLOCATION_RDP              = f'{ALLOCATION} (Our - {RDP})'
ALLOCATION_LOOSE_RDP        = f'{ALLOCATION} (DCO25 - {RDP})'
ALLOCATION_DECOMPOSITION    = f'{ALLOCATION} (Our - {DECOMPOSITION})'

# ======================= Methods Features =======================
@dataclass
class MethodFeatures:
    """
    Container for all features associated with a method.
    """
    name: str
    epsilon_calculator: Callable
    delta_calculator: Callable
    legend: str
    marker: str
    color: str

methods_dict = {
    LOCAL: MethodFeatures(
        name=LOCAL,
        epsilon_calculator=local_epsilon,
        delta_calculator=local_delta,
        legend='_{\\mathcal{L}}$ - ' + LOCAL,
        marker='*',
        color=colors_dict[LOCAL]
    ),
    POISSON_PLD: MethodFeatures(
        name=POISSON_PLD,
        epsilon_calculator=poisson_epsilon_pld,
        delta_calculator=poisson_delta_pld,
        legend='_{\\mathcal{P}}$ - ' + POISSON_PLD,
        marker='x',
        color=colors_dict[POISSON]
    ),
    POISSON_RDP: MethodFeatures(
        name=POISSON_RDP,
        epsilon_calculator=poisson_epsilon_rdp,
        delta_calculator=poisson_delta_rdp,
        legend='_{\\mathcal{P}}$ - ' + POISSON_RDP,
        marker='v',
        color=colors_dict[POISSON]
    ),
    SHUFFLE: MethodFeatures(
        name=SHUFFLE,
        epsilon_calculator=shuffle_epsilon_analytic,
        delta_calculator=shuffle_delta_analytic,
        legend='_{\\mathcal{S}}$ - ' + SHUFFLE,
        marker='p',
        color=colors_dict[SHUFFLE]
    ),
    ALLOCATION_ANALYTIC: MethodFeatures(
        name=ALLOCATION_ANALYTIC,
        epsilon_calculator=allocation_epsilon_analytic,
        delta_calculator=allocation_delta_analytic,
        legend='_{\\mathcal{A}}$ - ' + ALLOCATION_ANALYTIC,
        marker='P',
        color=colors_dict[ALLOCATION]
    ),
    ALLOCATION_RDP: MethodFeatures(
        name=ALLOCATION_RDP,
        epsilon_calculator=allocation_epsilon_rdp,
        delta_calculator=allocation_delta_rdp,
        legend='_{\\mathcal{A}}$ - ' + ALLOCATION_RDP,
        marker='^',
        color=colors_dict[ALLOCATION]
    ),
    ALLOCATION_LOOSE_RDP: MethodFeatures(
        name=ALLOCATION_LOOSE_RDP,
        epsilon_calculator=allocation_epsilon_rdp_loose,
        delta_calculator=allocation_delta_rdp_loose,
        legend='_{\\mathcal{A}}$ - ' + ALLOCATION_LOOSE_RDP,
        marker='o',
        color=colors_dict[ALLOCATION]
    ),
    ALLOCATION_DECOMPOSITION: MethodFeatures(
        name=ALLOCATION_DECOMPOSITION,
        epsilon_calculator=allocation_epsilon_decomposition,
        delta_calculator=allocation_delta_decomposition,
        legend='_{\\mathcal{A}}$ - ' + ALLOCATION_DECOMPOSITION,
        marker='X',
        color=colors_dict[ALLOCATION]
    ),
}

def get_features_for_methods(methods: List[str], feature: str) -> Dict[str, Any]:
    """
    Extract a specific feature for a list of methods using the global methods_dict.
    
    Args:
        methods: List of method keys
        feature: Name of the feature to extract
        
    Returns:
        Dictionary mapping method names to their feature values
    """
    try:
        return {method: getattr(methods_dict[method], feature) for method in methods}
    except KeyError as e:
        raise KeyError(f"Invalid method key: {e}")
    except AttributeError as e:
        raise AttributeError(f"Invalid feature name: {feature}")