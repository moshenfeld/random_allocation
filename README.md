# Random Allocation for Differential Privacy

This package implements random allocation mechanisms for differential privacy, providing various methods for privacy analysis including RDP, analytic, and decomposition approaches.

## Installation

### Using conda (recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/random_allocation.git
cd random_allocation

# Create and activate the conda environment
conda env create -f config/environment.yml
conda activate random_allocation

# Install the package in development mode
pip install -e .
```

### Using pip
```bash
pip install random_allocation
```

## Usage

```python
from random_allocation.comparisons.definitions import *
from random_allocation.comparisons.experiments import calc_params
from random_allocation.comparisons.visualization import plot_combined_data

# Example usage
params_dict = {
    'x_var': SIGMA,
    'y_var': EPSILON,
    SIGMA: np.exp(np.linspace(np.log(0.2), np.log(5), 20)),
    DELTA: 1e-10,
    NUM_STEPS: 100_000,
    NUM_SELECTED: 1,
    NUM_EPOCHS: 1
}

config_dict = {
    DISCRETIZATION: 1e-4,
    MIN_ALPHA: 2,
    MAX_ALPHA: 60
}

methods_list = [LOCAL, POISSON_PLD, SHUFFLE, ALLOCATION_RDP, ALLOCATION_ANALYTIC, ALLOCATION_DECOMPOSITION]

experiment_data = calc_params(params_dict, config_dict, methods_list)
plot_combined_data(experiment_data, log_x_axis=True, log_y_axis=True)
```

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