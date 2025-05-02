from typing import Dict, Any, List

from random_allocation.comparisons.structs import MethodFeatures
from random_allocation.other_schemes.local import *
from random_allocation.other_schemes.poisson import *
from random_allocation.other_schemes.shuffle import *
from random_allocation.random_allocation_scheme import *
"""
Common definitions for privacy parameters, scheme configurations, and experiment configuration.
"""

#======================= Direction =======================
ADD    = 'add'
REMOVE = 'remove'
BOTH   = 'both'

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

# ======================= Schemes =======================
LOCAL = 'Local'
POISSON = 'Poisson'
ALLOCATION = 'Allocation'
SHUFFLE = 'Shuffle'

colors_dict = {LOCAL: '#FF0000', POISSON: '#2BB22C', ALLOCATION: '#157DED', SHUFFLE: '#FF00FF'}

# ======================= Computation =======================
ANALYTIC = 'Analytic'
MONTE_CARLO = 'Monte Carlo'
PLD = 'PLD'
RDP = 'RDP'
DECOMPOSITION = 'Decomposition'
INVERSE = 'Inverse'
COMBINED = 'Combined'
RECURSIVE = 'Recursive'
LOWER_BOUND = 'Lower Bound'

# ======================= Methods =======================
POISSON_PLD                 = f'{POISSON} ({PLD})'
POISSON_RDP                 = f'{POISSON} ({RDP})'
ALLOCATION_ANALYTIC         = f'{ALLOCATION} (Our - {ANALYTIC})'
ALLOCATION_DIRECT           = f'{ALLOCATION} (Our - Direct)'
ALLOCATION_RDP_DCO          = f'{ALLOCATION} (DCO25 - {RDP})'
ALLOCATION_DECOMPOSITION    = f'{ALLOCATION} (Our - {DECOMPOSITION})'
ALLOCATION_COMBINED         = f'{ALLOCATION} (Our - {COMBINED})'
ALLOCATION_RECURSIVE        = f'{ALLOCATION} (Our - {RECURSIVE})'
ALLOCATION_MONTE_CARLO      = f'{ALLOCATION} (CGHKKLMSZ24 - {MONTE_CARLO})'
ALLOCATION_LOWER_BOUND      = f'{ALLOCATION} (CGHKKLMSZ24 - {LOWER_BOUND})'


# ======================= Methods Features =======================

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
        epsilon_calculator=Poisson_epsilon_PLD,
        delta_calculator=Poisson_delta_PLD,
        legend='_{\\mathcal{P}}$ - ' + POISSON_PLD,
        marker='x',
        color=colors_dict[POISSON]
    ),
    POISSON_RDP: MethodFeatures(
        name=POISSON_RDP,
        epsilon_calculator=Poisson_epsilon_RDP,
        delta_calculator=Poisson_delta_RDP,
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
    ALLOCATION_DIRECT: MethodFeatures(
        name=ALLOCATION_DIRECT,
        epsilon_calculator=allocation_epsilon_direct,
        delta_calculator=allocation_delta_direct,
        legend='_{\\mathcal{A}}$ - ' + ALLOCATION_DIRECT,
        marker='^',
        color=colors_dict[ALLOCATION]
    ),
    ALLOCATION_RDP_DCO: MethodFeatures(
        name=ALLOCATION_RDP_DCO,
        epsilon_calculator=allocation_epsilon_RDP_DCO,
        delta_calculator=allocation_delta_RDP_DCO,
        legend='_{\\mathcal{A}}$ - ' + ALLOCATION_RDP_DCO,
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
    ALLOCATION_COMBINED: MethodFeatures(
        name=ALLOCATION_COMBINED,
        epsilon_calculator=allocation_epsilon_combined,
        delta_calculator=allocation_delta_combined,
        legend='_{\\mathcal{A}}$ - ' + ALLOCATION_COMBINED,
        marker='s',
        color=colors_dict[ALLOCATION]
    ),
    ALLOCATION_RECURSIVE: MethodFeatures(
        name=ALLOCATION_RECURSIVE,
        epsilon_calculator=allocation_epsilon_recursive,
        delta_calculator=allocation_delta_recursive,
        legend='_{\\mathcal{A}}$ - ' + ALLOCATION_RECURSIVE,
        marker='h',
        color=colors_dict[ALLOCATION]
    ),
    ALLOCATION_MONTE_CARLO: MethodFeatures(
        name=ALLOCATION_MONTE_CARLO,
        epsilon_calculator=None,
        delta_calculator=allocation_delta_MC,
        legend='_{\\mathcal{A}}$ - ' + ALLOCATION_MONTE_CARLO,
        marker='D',
        color=colors_dict[ALLOCATION]
    ),
    ALLOCATION_LOWER_BOUND: MethodFeatures(
        name=ALLOCATION_LOWER_BOUND,
        epsilon_calculator=None,
        delta_calculator=allocation_delta_lower_bound,
        legend='_{\\mathcal{A}}$ - ' + ALLOCATION_LOWER_BOUND,
        marker='d',
        color=colors_dict[ALLOCATION]
    )
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