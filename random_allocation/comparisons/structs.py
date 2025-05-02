from dataclasses import dataclass
from typing import Callable, Optional, Literal
import numpy as np

"""
Common definitions for privacy parameters, scheme configurations, and experiment configuration.
"""

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
    MC_use_order_stats: bool = True
    MC_use_mean: bool = False
    MC_conf_level: float = 0.99
    MC_sample_size: int = 500_000

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