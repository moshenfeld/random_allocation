# Random Allocation for Differential Privacy

This package provides tools for analyzing and comparing different random allocation schemes in the context of differential privacy.

## Installation

You can install the package using pip:

```bash
pip install random-allocation
```

## Usage

Here's a simple example of how to use the package:

```python
from random_allocation.comparisons.definitions import PrivacyParams, SchemeConfig, Direction
from random_allocation.comparisons.definitions import (
    ALLOCATION_DIRECT, ALLOCATION_DECOMPOSITION, POISSON_PLD, LOCAL
)
from random_allocation.comparisons.experiments import run_experiment, PlotType
import numpy as np

# Define privacy parameters
params = {
    'x_var': 'sigma',            # Variable to vary on x-axis
    'y_var': 'epsilon',          # Variable to compute on y-axis
    'sigma': np.linspace(0.1, 2, 20),  # Range of sigma values to test
    'delta': 1e-6,               # Target delta
    'num_steps': 10000,          # Number of steps per epoch
    'num_selected': 1,           # Number of times each element is used
    'num_epochs': 1              # Number of epochs
}

# Define scheme configuration
config = SchemeConfig(
    allocation_direct_alpha_orders=[int(i) for i in np.arange(2, 61, dtype=int)]
)

# Define methods to compare
methods = [LOCAL, POISSON_PLD, ALLOCATION_DIRECT, ALLOCATION_DECOMPOSITION]

# Define visualization configuration
visualization_config = {
    'log_x_axis': True, 
    'log_y_axis': True,
    'format_x': lambda x, _: f'{x:.2f}'
}

# Run the experiment
results = run_experiment(
    params_dict=params,
    config=config, 
    methods=methods,
    visualization_config=visualization_config,
    experiment_name='epsilon_vs_sigma',
    plot_type=PlotType.COMBINED,
    read_data=False,           # Whether to try reading existing data first
    save_data=True,            # Whether to save computed data
    save_plots=True,           # Whether to save plots to files
    show_plots=True,           # Whether to display plots interactively
    direction=Direction.BOTH   # Analyze both add and remove directions
)
```

## Structure Overview

### Privacy Parameters

The `PrivacyParams` class encapsulates all parameters needed for privacy calculations:

```python
example_params = PrivacyParams(
    sigma        = 1.0,   # The Gaussian mechanism's noise scale
    num_steps    = 1_000, # The number of steps in each epoch of the scheme
    num_selected = 1,     # The number times each element is used in an epoch in the random allocation scheme.
                          # In the case of the Poisson scheme, sampling_probability=num_selected/num_steps
    num_epochs   = 1,     # The number of epochs the scheme is ran
    epsilon      = None,  # The target epsilon.
    delta        = 1e-5,  # The target delta.
                          # Exactly one value in {delta, epsilon} should be set to None
)
```

### Scheme Configuration

The `SchemeConfig` class provides configuration options for various privacy schemes:

```python
example_config = SchemeConfig(
    discretization = 1e-4,                             # The resolution used in the Gaussian's PLD numerical calculation.
                                                       # Default value = 1e-4.
    allocation_direct_alpha_orders = np.arange(2, 61), # The range of integer alpha orders used in the direct method for calculating the remove direction.
                                                       # Default value = None, and it must be defined.
    allocation_RDP_DCO_alpha_orders = None,            # The range of alpha orders used in the RDP DCO method.
                                                       # Default value = None, in which case default range of values is used.
    Poisson_alpha_orders = None,                       # The range of alpha orders used in the Poisson RDP method.
                                                       # Default value = None, in which case default range of values is used.
    print_alpha = False,                               # Indicates whether the alpha value used to set the optimal epsilon/delta value will be printed.
                                                       # Default value = False.
    delta_tolerance = 1e-15,                           # The desired resolution of the delta estimation, when using optimization based estimation methods.
                                                       # Default value = 1e-15.
    epsilon_tolerance = 1e-3,                          # The desired resolution of the epsilon estimation, when using optimization based estimation methods.
                                                       # Default value = 1e-3.
    epsilon_upper_bound = 100.0,                       # An upper bound on the search range of the epsilon estimation, when using binary search estimation methods.
                                                       # Default value = 100.0.
    MC_use_order_stats = True,                         # Indicates whether the order statistic technique should be used in the Monte Carlo simulation.
                                                       # Default value = True.
    MC_use_mean = False,                               # Indicates whether the Monte Carlo simulation method reports the mean or an upper bound.
                                                       # Default value = False, that is - report an upper bound.
    MC_conf_level = 0.99,                              # The probability with which the upper bound provided by the Monte Carlo simulation method is correct.
                                                       # Default value = 0.99.
    MC_sample_size = 500_000,                          # The sample size used in the Monte Carlo estimation.
                                                       # Default value = 500_000.
    shuffle_step = 100.0                               # Number of steps in the binary search performed in the shuffle code.
                                                       # Default value = 100.
)
```

### Direction Enum

The `Direction` enum specifies the direction of privacy analysis:

```python
from random_allocation.comparisons.definitions import Direction

# Available directions
Direction.ADD     # Add direction (adding a user to the dataset)
Direction.REMOVE  # Remove direction (removing a user from the dataset)
Direction.BOTH    # Both directions (consider both add and remove)
```

## Available Methods

The package includes several methods for privacy analysis:

### Local Methods

- `LOCAL`: Basic local mechanism with no randomization in the sampling.

### Poisson Methods

- `POISSON_PLD`: Poisson subsampling with Privacy Loss Distribution (PLD) analysis.
- `POISSON_RDP`: Poisson subsampling with Rényi Differential Privacy (RDP) analysis.

### Shuffle Methods

- `SHUFFLE`: Shuffle model of differential privacy.

### Random Allocation Methods

- `ALLOCATION_ANALYTIC`: Analytic method for the random allocation scheme.
- `ALLOCATION_DIRECT`: Direct computation method using Rényi Differential Privacy.
- `ALLOCATION_RDP_DCO`: RDP analysis following the approach of DCO25.
- `ALLOCATION_DECOMPOSITION`: Decomposition method for random allocation scheme.
- `ALLOCATION_COMBINED`: Combined approach leveraging multiple methods.
- `ALLOCATION_RECURSIVE`: Recursive computation approach.
- `ALLOCATION_MONTE_CARLO`: Monte Carlo simulation-based analysis (CGHKKLMSZ24).
- `ALLOCATION_LOWER_BOUND`: Lower bound analysis (CGHKKLMSZ24).

## Data Handling

The `run_experiment` function provides control over how data is read and saved:

```python
run_experiment(
    # Required parameters
    params_dict=params,            # Dictionary of privacy parameters
    config=config,                 # SchemeConfig object
    methods=methods,               # List of methods to compare
    
    # Optional parameters
    visualization_config=viz_config,  # Visualization settings
    experiment_name='my_experiment',  # Name used for saved files
    plot_type=PlotType.COMPARISON,    # COMPARISON or COMBINED
    
    # Data handling options
    read_data=False,        # Try reading existing data before calculating
    save_data=True,         # Save computed data to CSV files
    save_plots=True,        # Save plots to files
    show_plots=True,        # Display plots interactively
    
    # Privacy analysis direction
    direction=Direction.BOTH  # ADD, REMOVE, or BOTH
)
```

Using named parameters for all function arguments is recommended to avoid confusion about parameter order.

When `read_data=True`, the system will first attempt to load previously saved data from the `examples/data/` directory. If the data doesn't exist or `read_data=False`, new calculations will be performed.

## Plotting

The package supports two types of plots:
- `PlotType.COMPARISON`: For comparing different methods.
- `PlotType.COMBINED`: For showing combined results.

Visualization configuration options include:
- `log_x_axis`: Whether to use logarithmic scale for x-axis.
- `log_y_axis`: Whether to use logarithmic scale for y-axis.
- `format_x`: Function to format x-axis labels.
- `format_y`: Function to format y-axis labels.
- `legend_position`: Optional custom position for the legend (e.g., 'upper right', 'lower left').
- `num_y_ticks`: Optional number of y-axis tick marks to display (if not specified, matplotlib's default is used).

The visualization module includes intelligent legend positioning that automatically places legends to avoid overlapping with data points. If not specified, the system analyzes data distribution and places the legend in the area with the fewest data points.

When using logarithmic y-axis scaling (`log_y_axis=True`), the system automatically adjusts tick formatting for privacy parameters:
- For epsilon (ε) and delta (δ) parameters, small values (< 0.01) are displayed in scientific notation
- For other values, regular decimal formatting is used

For multi-plot figures, use the `plot_multiple_data` function:

```python
from random_allocation.comparisons.visualization import plot_multiple_data

# Create a multi-subplot figure with 3 plots in a row
fig = plot_multiple_data(
    data_list=[data1, data2, data3],  # List of data dictionaries
    titles=["Plot 1", "Plot 2", "Plot 3"],  # Optional titles for each subplot
    log_x_axis=True,
    log_y_axis=False,
    format_x=lambda x, _: f'{x:.2f}',
    plot_type='combined',  # 'combined' or 'comparison'
    figsize=(20, 6),  # Width, height in inches
    grid_layout=(1, 3),  # 1 row, 3 columns
    legend_position='upper right',  # Optional: specify legend position
    num_y_ticks=5  # Optional: control number of y-axis ticks
)

# Add a super title and adjust layout
plt.suptitle("My Multiple Plot Analysis", fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
```

## Testing

The project includes a test suite to ensure code functionality is maintained during refactoring. The tests are located in the `tests` directory and can be run using:

```bash
# Run all tests
python -m unittest discover tests

# Run a specific test file
python -m unittest tests.basic_tests
```

For more information about the test suite, see [tests/README.md](tests/README.md).

## Examples

The `examples` directory contains several reference implementations:

- `paper_experiments.py`: Reproduces experiments from the paper.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:
```
@article{yourcitation,
  title={Your Paper Title},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```