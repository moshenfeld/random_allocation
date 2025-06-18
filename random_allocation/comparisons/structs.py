"""
Common definitions for privacy parameters, scheme configurations, and experiment configuration.
"""

# Standard library imports
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Literal, Union, Any, Dict, List, cast

# Third-party imports
import numpy as np



class Direction(Enum):
    """Enum for direction of privacy analysis"""
    ADD = 'add'
    REMOVE = 'remove'
    BOTH = 'both'

class Verbosity(Enum):
    """Enum for verbosity levels in calculation functions"""
    NONE = 'none'
    WARNINGS = 'warnings'
    ALL = 'all'

@dataclass
class PrivacyParams:
    """Parameters common to all privacy schemes"""
    sigma: float
    num_steps: int
    num_selected: int = 1
    num_epochs: int = 1
    sampling_probability: float = 1.0
    # Either epsilon or delta must be provided, the other one will be computed
    epsilon: Optional[float] = None
    delta: Optional[float] = None
    
    def __post_init__(self):
        """Convert values to appropriate types and validate"""
        # Ensure proper types
        self.sigma = float(self.sigma)
        self.num_steps = int(self.num_steps)
        self.num_selected = int(self.num_selected)
        self.num_epochs = int(self.num_epochs)
        
        if self.epsilon is not None:
            self.epsilon = float(self.epsilon)
        if self.delta is not None:
            self.delta = float(self.delta)
    
    def validate(self) -> None:
        """Validate that the parameters are correctly specified"""
        # Fundamental constraints that should be enforced by programming logic
        assert self.epsilon is None or self.delta is None, "Only one of epsilon or delta should be provided"
        assert not (self.epsilon is None and self.delta is None), "Either epsilon or delta must be provided"
        
        # Value constraints that should never be violated in normal usage
        assert self.sigma > 0, f"sigma must be positive, got {self.sigma}"
        assert self.num_steps > 0, f"num_steps must be positive, got {self.num_steps}"
        assert self.num_selected > 0, f"num_selected must be positive, got {self.num_selected}"
        assert self.num_selected <= self.num_steps, f"num_selected must not exceed num_steps, got {self.num_selected} > {self.num_steps}"
        assert self.num_epochs > 0, f"num_epochs must be positive, got {self.num_epochs}"
        assert self.sampling_probability == 1.0 or self.num_selected == 1, \
            "sampling_probability can only be less than 1.0 if num_selected is 1"
        
        # Parameter value constraints
        if self.epsilon is not None:
            assert self.epsilon > 0, f"epsilon must be positive, got {self.epsilon}"
        if self.delta is not None:
            assert 0 < self.delta < 1, f"delta must be between 0 and 1, got {self.delta}"

@dataclass
class SchemeConfig:
    """Configuration for privacy schemes"""
    discretization: float = 1e-4
    allocation_direct_alpha_orders: Optional[List[int]] = None  # Will be set in __post_init__
    allocation_RDP_DCO_alpha_orders: Optional[List[float]] = None  # Will be set in __post_init__
    Poisson_alpha_orders: Optional[List[float]] = None  # Will be set in __post_init__
    delta_tolerance: float = 1e-15
    epsilon_tolerance: float = 1e-3
    epsilon_upper_bound: float = 100.0
    MC_use_order_stats: bool = True
    MC_use_mean: bool = False
    MC_conf_level: float = 0.99
    MC_sample_size: int = 500_000
    shuffle_step: float = 100.0
    verbosity: Verbosity = Verbosity.WARNINGS

@dataclass
class MethodFeatures:
    """
    Container for all features associated with a method.
    """
    name: str
    epsilon_calculator: Optional[Callable[[Any, Any], float]]
    delta_calculator: Optional[Callable[[Any, Any], float]]
    legend: str
    marker: str
    color: str