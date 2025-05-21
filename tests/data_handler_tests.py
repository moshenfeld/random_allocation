#!/usr/bin/env python
"""
Tests for the data handling functionality.

This script tests:
1. Saving experiment data in JSON and CSV formats
2. Loading experiment data from both formats
3. Ensuring all data is properly preserved
"""

import os
import sys
import unittest
import tempfile
import shutil
import glob
import numpy as np
import warnings

# Ignore warnings about non-interactive backend
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive, and thus cannot be shown")

# Ensure the project root is in the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from random_allocation.comparisons.data_handler import save_experiment_data, load_experiment_data
from random_allocation.comparisons.structs import SchemeConfig, Direction, PrivacyParams
from random_allocation.comparisons.experiments import run_experiment, PlotType
from random_allocation.comparisons.definitions import (
    EPSILON, DELTA, SIGMA, NUM_STEPS, NUM_SELECTED, NUM_EPOCHS,
    POISSON_RDP, ALLOCATION_DIRECT
)


class TestDataHandler(unittest.TestCase):
    """Test the data handling functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level resources and clean up any leftover test files"""
        cls.examples_dir = os.path.join(project_root, 'random_allocation', 'examples')
        cls.data_dir = os.path.join(cls.examples_dir, 'data')
        cls.plots_dir = os.path.join(cls.examples_dir, 'plots')
        
        # Clean up any test files from previous runs
        cls.cleanup_example_test_files()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up any test files created during testing"""
        cls.cleanup_example_test_files()
        
    @classmethod
    def cleanup_example_test_files(cls):
        """Remove all test files from the examples directory"""
        # Clean up data files
        test_data_patterns = [
            os.path.join(cls.data_dir, 'test_*.json'),
            os.path.join(cls.data_dir, 'test_*[!.json]'),
            os.path.join(cls.plots_dir, 'test_*_plot.png')
        ]
        
        for pattern in test_data_patterns:
            for file_path in glob.glob(pattern):
                try:
                    os.remove(file_path)
                    print(f"Removed test file: {file_path}")
                except (OSError, PermissionError) as e:
                    print(f"Failed to remove {file_path}: {e}")
    
    def setUp(self):
        """Set up temporary directories for testing"""
        self.test_dir = tempfile.mkdtemp()
        self.temp_data_dir = os.path.join(self.test_dir, "data")
        self.temp_plots_dir = os.path.join(self.test_dir, "plots")
        os.makedirs(self.temp_data_dir, exist_ok=True)
        os.makedirs(self.temp_plots_dir, exist_ok=True)
        
        # Create test data
        self.methods = [POISSON_RDP, ALLOCATION_DIRECT]
        self.test_data = {
            'x data': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            'y data': {
                POISSON_RDP: np.array([1.1, 1.2, 1.3, 1.4, 1.5]),
                ALLOCATION_DIRECT: np.array([2.1, 2.2, 2.3, 2.4, 2.5]),
                f"{POISSON_RDP}- std": np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
            },
            'x name': 'x axis',
            'y name': 'y axis',
            'title': 'Test plot',
            'sigma': 0.5,
            'delta': 1e-8,
            'num_steps': 1000,
            'num_selected': 1,
            'num_epochs': 1,
            'custom_field': 'custom value'
        }
        
    def tearDown(self):
        """Clean up temporary directories"""
        shutil.rmtree(self.test_dir)
        
    def test_save_load_direct(self):
        """Test saving and loading data directly"""
        # Paths for test files
        json_path = os.path.join(self.temp_data_dir, "test_data")
        
        # Test saving
        save_experiment_data(self.test_data, self.methods, json_path)
        
        # Check that files exist
        self.assertTrue(os.path.exists(f"{json_path}.json"), "JSON file was not created")
        self.assertTrue(os.path.exists(json_path), "CSV file was not created")
        
        # Test loading from JSON
        loaded_data = load_experiment_data(json_path, self.methods)
        self.assertIsNotNone(loaded_data, "Failed to load data from JSON")
        
        # Check that all data is preserved
        for key in self.test_data:
            self.assertIn(key, loaded_data, f"Key {key} is missing from loaded data")
            
            if key == 'y data':
                for method in self.test_data[key]:
                    self.assertIn(method, loaded_data[key], f"Method {method} is missing from loaded y data")
                    np.testing.assert_array_almost_equal(
                        self.test_data[key][method], 
                        loaded_data[key][method],
                        err_msg=f"Y data for {method} doesn't match"
                    )
            elif isinstance(self.test_data[key], np.ndarray):
                np.testing.assert_array_almost_equal(
                    self.test_data[key], 
                    loaded_data[key],
                    err_msg=f"Array data for {key} doesn't match"
                )
            else:
                self.assertEqual(self.test_data[key], loaded_data[key], f"Data for {key} doesn't match")
                
        # Delete JSON file and test loading from CSV
        os.remove(f"{json_path}.json")
        loaded_data_csv = load_experiment_data(json_path, self.methods)
        self.assertIsNotNone(loaded_data_csv, "Failed to load data from CSV")
        
        # Check core data from CSV
        np.testing.assert_array_almost_equal(
            self.test_data['x data'], 
            loaded_data_csv['x data'],
            err_msg="X data doesn't match from CSV"
        )
        for method in self.methods:
            if method in self.test_data['y data']:
                np.testing.assert_array_almost_equal(
                    self.test_data['y data'][method], 
                    loaded_data_csv['y data'][method],
                    err_msg=f"Y data for {method} doesn't match from CSV"
                )
                
    def test_run_experiment_save_load(self):
        """Test saving and loading data with run_experiment"""
        # Configure experiment
        params_dict = {
            'x_var': SIGMA,
            'y_var': EPSILON,
            SIGMA: np.linspace(0.2, 2.0, 3),  # Small number of points for faster testing
            DELTA: 1e-8,
            NUM_STEPS: 1000,
            NUM_SELECTED: 1,
            NUM_EPOCHS: 1
        }
        
        config = SchemeConfig(allocation_direct_alpha_orders=[int(i) for i in np.arange(2, 11, dtype=int)])
        visualization_config = {'log_x_axis': True, 'log_y_axis': True}
        experiment_name = 'test_run_experiment'
        
        # Instead of patching paths, we'll create our own save/load functions that use our temp directory
        from random_allocation.comparisons import experiments
        original_save_experiment_data = experiments.save_experiment_data
        original_load_experiment_data = experiments.load_experiment_data
        
        # Create patched functions
        def patched_save_experiment_data(data, methods, experiment_name):
            # Replace the path to use our test directory
            base_name = os.path.basename(experiment_name)
            new_path = os.path.join(self.temp_data_dir, base_name)
            return original_save_experiment_data(data, methods, new_path)
            
        def patched_load_experiment_data(experiment_name, methods):
            # Replace the path to use our test directory
            base_name = os.path.basename(experiment_name)
            new_path = os.path.join(self.temp_data_dir, base_name)
            return original_load_experiment_data(new_path, methods)
        
        # Apply the patches
        experiments.save_experiment_data = patched_save_experiment_data
        experiments.load_experiment_data = patched_load_experiment_data
        
        try:
            # First run - compute and save
            data_1 = run_experiment(
                params_dict.copy(), 
                config, 
                self.methods, 
                visualization_config, 
                experiment_name, 
                PlotType.COMPARISON, 
                save_data=True, 
                save_plots=False,  # Don't save plots to keep test focused
                direction=Direction.BOTH
            )
            
            # Check that files were created
            data_file = os.path.join(self.temp_data_dir, experiment_name)
            self.assertTrue(os.path.exists(f"{data_file}.json"), "JSON file was not created")
            self.assertTrue(os.path.exists(data_file), "CSV file was not created")
            
            # Second run - load from saved data
            data_2 = run_experiment(
                params_dict.copy(),
                config, 
                self.methods, 
                visualization_config, 
                experiment_name, 
                PlotType.COMPARISON, 
                save_data=False, 
                save_plots=False,  # Don't save plots
                direction=Direction.BOTH
            )
            
            # Compare data dictionaries
            self.assertEqual(len(data_1['x data']), len(data_2['x data']), "X data length doesn't match")
            np.testing.assert_array_almost_equal(
                data_1['x data'], 
                data_2['x data'],
                err_msg="X data arrays don't match"
            )
            
            for method in self.methods:
                self.assertIn(method, data_1['y data'], f"Method {method} missing from first run data")
                self.assertIn(method, data_2['y data'], f"Method {method} missing from second run data")
                np.testing.assert_array_almost_equal(
                    data_1['y data'][method], 
                    data_2['y data'][method],
                    err_msg=f"Y data for {method} doesn't match between runs"
                )
                
        finally:
            # Restore original functions
            experiments.save_experiment_data = original_save_experiment_data
            experiments.load_experiment_data = original_load_experiment_data


def run_tests():
    """Run the tests"""
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestDataHandler)
    test_result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    return test_result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 