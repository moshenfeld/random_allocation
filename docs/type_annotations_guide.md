# Type Annotations Guide for Random Allocation

This document outlines the type annotation standards established for the Random Allocation project.

## Type Annotation Standards

### 1. Import Structure

```python
# Standard library imports
from typing import Dict, List, Optional, Union, Callable, Any, TypeVar, cast

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt

# Local application imports
from random_allocation.comparisons.definitions import PrivacyParams, SchemeConfig
```

### 2. Common Type Aliases

```python
# Type aliases
T = TypeVar('T')  # Generic type variable
DataDict = Dict[str, Any]  # Dictionary containing experiment data
MethodList = List[str]  # List of method names
XValues = List[Union[float, int]]  # List of x-axis values
FormatterFunc = Callable[[float, int], str]  # Function for formatting axis values
EpsilonCalculator = Callable[[PrivacyParams, SchemeConfig], float]  # Function for calculating epsilon
DeltaCalculator = Callable[[PrivacyParams, SchemeConfig], float]  # Function for calculating delta
```

### 3. Function Signatures

All functions should include parameter and return type annotations:

```python
def allocation_epsilon_analytic(params: PrivacyParams,
                              config: SchemeConfig,
                              ) -> Optional[float]:
    """
    Compute epsilon for the analytic allocation scheme.
    
    Args:
        params: Privacy parameters
        config: Scheme configuration parameters
    
    Returns:
        Computed epsilon value or None if conditions are not met
    """
    # Function implementation
```

### 4. Class Attributes

```python
@dataclass
class PrivacyParams:
    """Parameters common to all privacy schemes"""
    sigma: float
    num_steps: int
    num_selected: int
    num_epochs: int
    epsilon: Optional[float] = None
    delta: Optional[float] = None
```

### 5. Variable Annotations

For variables within functions, add type annotations where helpful:

```python
# Dictionary from method name to calculator function
func_dict: Dict[str, Optional[Callable]] = get_func_dict(methods, y_var)

# Array of results
results: List[float] = []
```

### 6. Constants

Constants should be annotated with their specific type:

```python
EPSILON: str = "epsilon"
VARIABLES: List[str] = [EPSILON, DELTA, SIGMA, NUM_STEPS, NUM_SELECTED, NUM_EPOCHS]
```

## Gradual Typing Strategy

This project uses a gradual typing approach:

1. Core data structures are fully typed
2. Public APIs have complete type annotations
3. Internal implementation details may have partial typing
4. The mypy configuration allows for incremental adoption

## Type Checking

Use mypy to check type consistency:

```bash
mypy random_allocation
```

The mypy configuration in pyproject.toml sets reasonable defaults for gradual typing.

### Integration with Test Suite

Type checking is integrated into the test suite at multiple levels:

```bash
# Basic type checking (development)
python tests/run_tests.py basic

# Comprehensive type validation (pre-release)
python tests/run_tests.py full

# Complete type annotation coverage (release)
python tests/run_tests.py release
```

The release-level tests include comprehensive type annotation validation covering:
- Type alias compliance
- Function signature completeness
- Runtime type validation
- Constant type annotations
- MyPy integration testing

### Current Type Coverage

The project maintains high type annotation coverage:
- Core data structures: 100% typed
- Public APIs: Complete type annotations
- Internal functions: Gradual typing approach
- Test suite: Validates type annotation compliance

All type annotations follow the standards outlined in this guide and are validated through automated testing.
