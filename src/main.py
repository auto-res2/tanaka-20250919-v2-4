import argparse
import yaml
import torch
import os
import json
import random
import numpy as np
from datetime import datetime

from . import preprocess
from . import train
from . import evaluate

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_experiment(config_path):
    """Loads config and runs a full experiment end-to-end."""
    print(f"Loading configuration from {config_path}...")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return

    print("Configuration loaded successfully.")
    print(json.dumps(config, indent=2))

    # Setup environment
    set_seed(config['training'].get('seed', 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create results directory
    results_dir = ".research/iteration1"
    os.makedirs(results_dir, exist_ok=True)

    # --- 1. Data Preprocessing ---
    print("\n--- Starting Data Preprocessing ---")
    train_loader, val_loader = preprocess.get_dataloaders(config)

    # --- 2. Model, Optimizer, and Loss Setup ---
    print("\n--- Setting up Models and Optimizer ---")
    model, base_model, optimizer = train.setup_models_and_optimizer(config, device)

    # --- 3. Training ---
    print("\n--- Starting Training ---")
    trained_model = train.run_training_loop(config, model, base_model, train_loader, val_loader, optimizer, device)

    # --- 4. Evaluation ---
    print("\n--- Starting Evaluation ---")
    eval_results = evaluate.run_evaluation(trained_model, config, device)

    # --- 5. Save Results ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = os.path.join(results_dir, f"results_{timestamp}.json")
    
    final_output = {
        'config': config,
        'evaluation_results': eval_results
    }
    
    try:
        with open(result_filename, 'w') as f:
            json.dump(final_output, f, indent=2)
        print(f"\nResults saved to {result_filename}")
        print("--- Final Results JSON ---")
        print(json.dumps(final_output, indent=2))
    except IOError as e:
        print(f"Error saving results to file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run AIBL Experiment")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--smoke-test', action='store_true', help='Run a small-scale smoke test.')
    group.add_argument('--full-experiment', action='store_true', help='Run the full experiment.')

    args = parser.parse_args()

    if args.smoke_test:
        config_file = 'config/smoke_test.yaml'
        print("--- Running Smoke Test ---")
        run_experiment(config_file)
        print("--- Smoke Test Passed ---")
    elif args.full_experiment:
        config_file = 'config/full_experiment.yaml'
        print("--- Running Full Experiment ---")
        run_experiment(config_file)

if __name__ == "__main__":
    main()
