from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional, Literal
import numpy as np

"""
Common definitions for privacy parameters, scheme configurations, and experiment configuration.
"""

#======================= Privacy Parameters & Scheme Config =======================
@dataclass
class PrivacyParams:
    """Parameters common to all privacy schemes"""
    sigma: float
    num_steps: int
    num_selected: int
    num_epochs: int
    # Either epsilon or delta must be provided, the other one will be computed
    epsilon: Optional[float] = None
    delta: Optional[float] = None
    
    def validate(self):
        """Validate that the parameters are correctly specified"""
        if self.epsilon is None and self.delta is None:
            raise ValueError("Either epsilon or delta must be provided")
        if self.epsilon is not None and self.delta is not None:
            raise ValueError("Only one of epsilon or delta should be provided")

@dataclass
class SchemeConfig:
    """Configuration for privacy schemes"""
    direction: Literal['add', 'remove', 'both'] = 'both'
    discretization: float = 1e-4
    allocation_direct_alpha_orders: np.ndarray = None  # Will be set in __post_init__
    allocation_RDP_DCO_alpha_orders: np.ndarray = None  # Will be set in __post_init__
    Poisson_alpha_orders: np.ndarray = None  # Will be set in __post_init__
    print_alpha: bool = False
    delta_tolerance: float = 1e-15
    epsilon_tolerance: float = 1e-3
    epsilon_upper_bound: float = 100.0
    
    def __post_init__(self):
        """Initialize alpha_orders if not provided"""
        if self.allocation_direct_alpha_orders is None:
            min_alpha = 2
            max_alpha = 50
            self.allocation_direct_alpha_orders = np.arange(min_alpha, max_alpha + 1, 1)

# Import privacy scheme functions after defining the dataclasses they need
from random_allocation.other_schemes.local import local_epsilon, local_delta
from random_allocation.other_schemes.poisson import Poisson_epsilon_PLD, Poisson_delta_PLD, Poisson_epsilon_RDP, Poisson_delta_RDP
from random_allocation.other_schemes.shuffle import shuffle_epsilon_analytic, shuffle_delta_analytic

from random_allocation.random_allocation_scheme import allocation_epsilon_analytic, allocation_delta_analytic
from random_allocation.random_allocation_scheme import allocation_epsilon_direct, allocation_delta_direct
from random_allocation.random_allocation_scheme import allocation_epsilon_RDP_DCO, allocation_delta_RDP_DCO
from random_allocation.random_allocation_scheme import allocation_epsilon_decomposition, allocation_delta_decomposition
from random_allocation.random_allocation_scheme import allocation_epsilon_combined, allocation_delta_combined
from random_allocation.random_allocation_scheme import allocation_epsilon_recursive, allocation_delta_recursive

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

#===================== Configuration =====================
NUM_EXP           = 'num_experiments'
DISCRETIZATION    = 'discretization'
MIN_ALPHA         = 'min_alpha'  # Kept for backward compatibility
MAX_ALPHA         = 'max_alpha'  # Kept for backward compatibility
ALPHA_ORDERS      = 'allocation_direct_alpha_orders'
EPSILON_TOLERANCE = 'epsilon_tolerance'
DELTA_TOLERANCE   = 'delta_tolerance'
DIRECTION         = 'direction'
CONFIGS           = [NUM_EXP, DISCRETIZATION, ALPHA_ORDERS, EPSILON_TOLERANCE, DELTA_TOLERANCE, DIRECTION]

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
INVERSE = 'Inverse'
COMBINED = 'Combined'
RECURSIVE = 'Recursive'

# ======================= Methods =======================
POISSON_PLD                 = f'{POISSON} ({PLD})'
POISSON_RDP                 = f'{POISSON} ({RDP})'
ALLOCATION_ANALYTIC         = f'{ALLOCATION} (Our - {ANALYTIC})'
ALLOCATION_RDP              = f'{ALLOCATION} (Our - {RDP})'
ALLOCATION_RDP_DCO          = f'{ALLOCATION} (DCO25 - {RDP})'
ALLOCATION_DECOMPOSITION    = f'{ALLOCATION} (Our - {DECOMPOSITION})'
ALLOCATION_COMBINED         = f'{ALLOCATION} (Our - {COMBINED})'
ALLOCATION_RECURSIVE         = f'{ALLOCATION} (Our - {RECURSIVE})'

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
    ALLOCATION_RDP: MethodFeatures(
        name=ALLOCATION_RDP,
        epsilon_calculator=allocation_epsilon_direct,
        delta_calculator=allocation_delta_direct,
        legend='_{\\mathcal{A}}$ - ' + ALLOCATION_RDP,
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