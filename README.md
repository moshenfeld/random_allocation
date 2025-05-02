# Random Allocation for Differential Privacy

This package provides tools for analyzing and comparing different random allocation schemes in the context of differential privacy.

## Installation

You can install the package using pip:

```bash
pip install random-allocation
```

## Usage

Here's a simple example of how to use the package to run experiments:

```python
from random_allocation import run_experiment, PlotType
from random_allocation import ALLOCATION, ALLOCATION_ANALYTIC, ALLOCATION_DIRECT, ALLOCATION_DECOMPOSITION

# Define experiment parameters
params_dict = {
    'x_var': 'sigma',
    'y_var': 'epsilon',
    'sigma': [0.1, 0.2, 0.3, 0.4, 0.5],
    'n': 1000,
    'k': 10,
    'delta': 1e-5
}

# Define configuration
config_dict = {
    'title': 'Sigma vs Epsilon',
    'x name': 'Sigma',
    'y name': 'Epsilon'
}

# Define visualization configuration
visualization_config = {
    'log_x_axis': False,
    'log_y_axis': True
}

# Define methods to compare
methods = [ALLOCATION_ANALYTIC, ALLOCATION_DIRECT, ALLOCATION_DECOMPOSITION]

# Run the experiment
run_experiment(
    params_dict=params_dict,
    config_dict=config_dict,
    methods=methods,
    visualization_config=visualization_config,
    experiment_name='sigma_vs_epsilon',
    plot_type=PlotType.COMPARISON,
    save_data=True,
    save_plots=True
)
```

## Creating Custom Experiments

To create your own experiments:

1. Create a new Python file (e.g., `my_experiments.py`)
2. Import the necessary functions and constants from `random_allocation`
3. Define your experiment parameters, configuration, and methods
4. Call `run_experiment` with your settings

The package provides two types of plots:
- `PlotType.COMPARISON`: For comparing different methods
- `PlotType.COMBINED`: For showing combined results

## Available Methods

The package includes several methods for comparison:
- `ALLOCATION_ANALYTIC`: Our analytic method
- `ALLOCATION_DIRECT`: Our RDP-based method
- `ALLOCATION_DECOMPOSITION`: Our decomposition method

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