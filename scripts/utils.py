# scripts/utils.py

import os
import pickle
import logging
from models.rbm import RBM

def setup_logging(log_file, project_root):
    """
    Configure logging to output to both console and a file.
    
    Parameters:
    - log_file (str): Name of the log file.
    - project_root (str): Absolute path to the project root directory.
    """
    log_path = os.path.join(project_root, log_file)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

def ensure_directory(path):
    """
    Ensure that a directory exists. If not, create it.
    
    Parameters:
    - path (str): Path to the directory.
    """
    os.makedirs(path, exist_ok=True)

def load_model(model_path, rbm, project_root):
    """
    Load RBM weights and biases from a saved model file.
    
    Parameters:
    - model_path (str): Relative path to the saved model file.
    - rbm (RBM): RBM instance to load parameters into.
    - project_root (str): Absolute path to the project root directory.
    
    Raises:
    - FileNotFoundError: If the model file does not exist.
    """
    absolute_model_path = os.path.join(project_root, model_path)
    
    if not os.path.exists(absolute_model_path):
        raise FileNotFoundError(f"Model file '{absolute_model_path}' does not exist.")
    
    with open(absolute_model_path, 'rb') as f:
        model_data = pickle.load(f)
        rbm.set_weights(model_data['weights'])
        rbm.set_visible_bias(model_data['visible_bias'])
        rbm.set_hidden_bias(model_data['hidden_bias'])
