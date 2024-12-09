# scripts/train.py

import sys
import os

# Without this project won't work, not sure what the proper fix is
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
import argparse
from models.rbm import RBM
from load_data import MnistDataloader
import logging
import matplotlib.pyplot as plt

def setup_logging(log_file='training.log'):
    """
    Configure logging to output to both console and a file.
    
    Parameters:
    - log_file (str): Path to the log file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def parse_arguments():
    """
    Parse command-line arguments for hyperparameters.
    
    Returns:
    - argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Train RBM on MNIST Dataset')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for weight updates')
    parser.add_argument('--momentum', type=float, default=0.5, help='Momentum for weight updates')
    parser.add_argument('--weight_decay', type=float, default=0.0002, help='Weight decay (L2 regularization)')
    parser.add_argument('--k', type=int, default=1, help='Number of Gibbs sampling steps')
    parser.add_argument('--is_binary', type=bool, default=True, help='Use binary visible units if True, else real-valued')
    parser.add_argument('--n_hidden', type=int, default=64, help='Number of hidden units')
    parser.add_argument('--batch_size', type=int, default=64, help='Mini-batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--early_stop', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--train_split', type=float, default=0.9, help='Proportion of training data for training vs validation')
    return parser.parse_args(args=[])

def ensure_directory(path):
    """
    Ensure that a directory exists. If not, create it.
    
    Parameters:
    - path (str): Path to the directory.
    """
    os.makedirs(path, exist_ok=True)

def load_data(is_binary, train_split=0.9):
    """
    Load and split the MNIST dataset into training and validation sets.
    
    Parameters:
    - is_binary (bool): Whether to preprocess data as binary.
    - train_split (float): Proportion of data to use for training.
    
    Returns:
    - tuple: (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)
    """
    # Define file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    base_path = os.path.join(project_root, 'data', 'raw')
    
    training_images_filepath = os.path.join(base_path, 'train-images.idx3-ubyte')
    training_labels_filepath = os.path.join(base_path, 'train-labels.idx1-ubyte')
    test_images_filepath = os.path.join(base_path, 't10k-images.idx3-ubyte')
    test_labels_filepath = os.path.join(base_path, 't10k-labels.idx1-ubyte')
    
    # Initialize data loader
    data_loader = MnistDataloader(
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
        is_binary=is_binary
    )
    
    # Load data
    (x_train_full, y_train_full), (x_test, y_test) = data_loader.load_data()
    
    # Split into training and validation
    split_idx = int(train_split * x_train_full.shape[0])
    x_train, y_train = x_train_full[:split_idx], y_train_full[:split_idx]
    x_val, y_val = x_train_full[split_idx:], y_train_full[split_idx:]
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def train_rbm(rbm, data, epochs, batch_size, early_stop, logger):
    """
    Train the RBM using Contrastive Divergence.
    
    Parameters:
    - rbm (RBM): The RBM model instance.
    - data (tuple): (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)
    - epochs (int): Number of training epochs.
    - batch_size (int): Size of each mini-batch.
    - early_stop (int): Patience for early stopping.
    - logger (logging.Logger): Logger instance.
    
    Returns:
    - RBM: Trained RBM model.
    """
    (x_train, y_train), (x_val, y_val), _ = data
    num_samples = x_train.shape[0]
    num_batches = num_samples // batch_size
    best_val_error = float('inf')
    patience_counter = 0
    
    # Ensure 'saved_models' directory exists
    ensure_directory('saved_models')
    
    for epoch in range(1, epochs + 1):
        # Shuffle training data
        permutation = np.random.permutation(num_samples)
        x_train_shuffled = x_train[permutation]
        y_train_shuffled = y_train[permutation]
        
        epoch_error = 0.0
        
        for batch_idx in range(num_batches):
            batch = x_train_shuffled[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            rbm.contrastive_divergence(batch)
            reconstructed = rbm.reconstruct(batch)
            batch_error = np.mean((batch - reconstructed) ** 2)
            epoch_error += batch_error
        
        avg_epoch_error = epoch_error / num_batches
        
        # Validate
        reconstructed_val = rbm.reconstruct(x_val)
        val_error = np.mean((x_val - reconstructed_val) ** 2)
        
        logger.info(f"Epoch {epoch}/{epochs} - Training Error: {avg_epoch_error:.4f} - Validation Error: {val_error:.4f}")
        
        # Early Stopping Check
        if val_error < best_val_error:
            best_val_error = val_error
            patience_counter = 0
            # Save the best model
            best_model_path = os.path.join('saved_models', 'rbm_best_model.pkl')
            with open(best_model_path, 'wb') as f:
                pickle.dump({
                    'weights': rbm.get_weights(),
                    'visible_bias': rbm.get_visible_bias(),
                    'hidden_bias': rbm.get_hidden_bias()
                }, f)
            logger.info("Validation error improved. Model saved.")
        else:
            patience_counter += 1
            logger.info(f"No improvement in validation error for {patience_counter} epochs.")
            if patience_counter >= early_stop:
                logger.info("Early stopping triggered.")
                break
    
    # Load the best model
    best_model_path = os.path.join('saved_models', 'rbm_best_model.pkl')
    with open(best_model_path, 'rb') as f:
        model_data = pickle.load(f)
        rbm.set_weights(model_data['weights'])
        rbm.set_visible_bias(model_data['visible_bias'])
        rbm.set_hidden_bias(model_data['hidden_bias'])
    
    return rbm

def main():
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("RBM Training Started.")
    logger.info(f"Hyperparameters: Learning Rate={args.learning_rate}, Momentum={args.momentum}, "
                f"Weight Decay={args.weight_decay}, k={args.k}, is_binary={args.is_binary}, "
                f"n_hidden={args.n_hidden}, Batch Size={args.batch_size}, Epochs={args.epochs}, "
                f"Early Stop={args.early_stop}")
    
    # Load data
    data = load_data(is_binary=args.is_binary, train_split=args.train_split)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data
    logger.info(f"Training Set Size: {x_train.shape[0]}")
    logger.info(f"Validation Set Size: {x_val.shape[0]}")
    logger.info(f"Test Set Size: {x_test.shape[0]}")

    # Initialize RBM
    rbm = RBM(
        n_visible=x_train.shape[1],
        n_hidden=args.n_hidden,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        use_pcd=False,
        k=args.k,
        visible_type='binary' if args.is_binary else 'real',
        batch_size=args.batch_size
    )
    logger.info("RBM initialized.")

    # Train RBM
    rbm = train_rbm(rbm, data, args.epochs, args.batch_size, args.early_stop, logger)
    logger.info("RBM Training Completed.")

    # Evaluate on Test Set
    reconstructed_test = rbm.reconstruct(x_test)
    test_reconstruction_error = np.mean((x_test - reconstructed_test) ** 2)
    logger.info(f"Test Reconstruction Error: {test_reconstruction_error:.4f}")

    # Save the final model
    final_model_path = os.path.join('saved_models', 'rbm_final_model.pkl')
    with open(final_model_path, 'wb') as f:
        pickle.dump({
            'weights': rbm.get_weights(),
            'visible_bias': rbm.get_visible_bias(),
            'hidden_bias': rbm.get_hidden_bias()
        }, f)
    logger.info("Final model saved to 'saved_models/rbm_final_model.pkl'.")

if __name__ == "__main__":
    main()
