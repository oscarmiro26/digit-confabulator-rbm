# scripts/train.py

import sys
import os
import numpy as np
import pickle
import logging
from itertools import product

# Define project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.rbm import RBM
from load_data import MnistDataloader
from utils import setup_logging, ensure_directory

# Core hyperparameters
LEARNING_RATE = 0.1
MOMENTUM = 0.5
WEIGHT_DECAY = 0.0002
K = 1
IS_BINARY = True
N_HIDDEN = 64
BATCH_SIZE = 64
EPOCHS = 50
EARLY_STOP = 5
TRAIN_SPLIT = 0.9

HYPEROPT = True

# Model save path
MODEL_SAVE_PATH = 'saved_models/rbm_final_model.pkl'


def load_data(is_binary, train_split=0.9):
    """
    Load and split the MNIST dataset into training and validation sets.
    """
    base_path = os.path.join(project_root, 'data', 'raw')

    training_images_filepath = os.path.join(base_path, 'train-images.idx3-ubyte')
    training_labels_filepath = os.path.join(base_path, 'train-labels.idx1-ubyte')
    test_images_filepath = os.path.join(base_path, 't10k-images.idx3-ubyte')
    test_labels_filepath = os.path.join(base_path, 't10k-labels.idx1-ubyte')

    validation_split = 1 - train_split
    data_loader = MnistDataloader(
        training_images_filepath=training_images_filepath,
        training_labels_filepath=training_labels_filepath,
        test_images_filepath=test_images_filepath,
        test_labels_filepath=test_labels_filepath,
        is_binary=is_binary,
        validation_split=validation_split
    )

    return data_loader.load_data()

def train_rbm(rbm, data, epochs, batch_size, early_stop, logger, saved_models_dir):
    """
    Train the RBM using Contrastive Divergence.
    """
    (x_train, y_train), (x_val, y_val), _ = data
    num_samples = x_train.shape[0]
    num_batches = num_samples // batch_size
    best_val_error = float('inf')
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        permutation = np.random.permutation(num_samples)
        x_train_shuffled = x_train[permutation]

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

        # Early Stopping
        if val_error < best_val_error:
            best_val_error = val_error
            patience_counter = 0
            # Save the best model
            best_model_path = os.path.join(saved_models_dir, 'rbm_best_model.pkl')
            with open(best_model_path, 'wb') as f:
                pickle.dump({
                    'weights': rbm.get_weights(),
                    'visible_bias': rbm.get_visible_bias(),
                    'hidden_bias': rbm.get_hidden_bias()
                }, f)
        else:
            patience_counter += 1
            if patience_counter >= early_stop:
                logger.info("Early stopping triggered.")
                break

    # Load the best model
    best_model_path = os.path.join(saved_models_dir, 'rbm_best_model.pkl')
    with open(best_model_path, 'rb') as f:
        model_data = pickle.load(f)
    rbm.set_weights(model_data['weights'])
    rbm.set_visible_bias(model_data['visible_bias'])
    rbm.set_hidden_bias(model_data['hidden_bias'])

    return rbm, best_val_error

def hyperparameter_search(data, logger, saved_models_dir):
    """
    Perform a simple hyperparameter search to find the best combination.
    We'll do a small grid search over a few parameters.
    """
    learning_rates = [0.1] # 0.005
    momenta = [0.9]
    weight_decays = [0.0001] # 0.0002
    ks = [10]
    n_hiddens = [128]

    best_config = None
    best_val_error = float('inf')

    for lr, mom, wd, k_val, nh in product(learning_rates, momenta, weight_decays, ks, n_hiddens):
        logger.info(f"Testing config: LR={lr}, MOM={mom}, WD={wd}, k={k_val}, n_hidden={nh}")

        rbm = RBM(
            n_visible=data[0][0].shape[1],
            n_hidden=nh,
            learning_rate=lr,
            momentum=mom,
            weight_decay=wd,
            use_pcd=False,
            k=k_val,
            is_binary=IS_BINARY,
            batch_size=BATCH_SIZE
        )

        _, val_error = train_rbm(rbm, data, EPOCHS, BATCH_SIZE, EARLY_STOP, logger, saved_models_dir)

        if val_error < best_val_error:
            best_val_error = val_error
            best_config = (lr, mom, wd, k_val, nh)
            logger.info(f"New best config: LR={lr}, MOM={mom}, WD={wd}, k={k_val}, n_hidden={nh}, Val Error={val_error:.4f}")

    logger.info(f"Best config found: LR={best_config[0]}, MOM={best_config[1]}, WD={best_config[2]}, k={best_config[3]}, n_hidden={best_config[4]} with Val Error={best_val_error:.4f}")
    return best_config

def main():
    # Setup logging
    setup_logging(log_file='training.log', project_root=project_root)
    logger = logging.getLogger(__name__)

    logger.info("RBM Training Started.")
    logger.info("Hyperparameters:")
    logger.info(f"LEARNING_RATE={LEARNING_RATE}, MOMENTUM={MOMENTUM}, WEIGHT_DECAY={WEIGHT_DECAY}, k={K}, "
                f"IS_BINARY={IS_BINARY}, N_HIDDEN={N_HIDDEN}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}, "
                f"EARLY_STOP={EARLY_STOP}, TRAIN_SPLIT={TRAIN_SPLIT}, HYPEROPT={HYPEROPT}")

    # Load data
    data = load_data(is_binary=IS_BINARY, train_split=TRAIN_SPLIT)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data
    logger.info(f"Training Set Size: {x_train.shape[0]}")
    logger.info(f"Validation Set Size: {x_val.shape[0]}")
    logger.info(f"Test Set Size: {x_test.shape[0]}")

    # Define the path for saving models at the project root
    saved_models_dir = os.path.join(project_root, 'saved_models')
    ensure_directory(saved_models_dir)

    current_lr = LEARNING_RATE
    current_momentum = MOMENTUM
    current_weight_decay = WEIGHT_DECAY
    current_k = K
    current_n_hidden = N_HIDDEN

    if HYPEROPT:
        # Perform hyperparameter search
        best_config = hyperparameter_search(data, logger, saved_models_dir)
        current_lr = best_config[0]
        current_momentum = best_config[1]
        current_weight_decay = best_config[2]
        current_k = best_config[3]
        current_n_hidden = best_config[4]

    # Initialize RBM with chosen hyperparameters
    rbm = RBM(
        n_visible=x_train.shape[1],
        n_hidden=current_n_hidden,
        learning_rate=current_lr,
        momentum=current_momentum,
        weight_decay=current_weight_decay,
        use_pcd=False,
        k=current_k,
        is_binary=IS_BINARY,
        batch_size=BATCH_SIZE
    )
    logger.info("RBM initialized with final hyperparameters.")

    # Train RBM
    rbm, val_error = train_rbm(rbm, data, EPOCHS, BATCH_SIZE, EARLY_STOP, logger, saved_models_dir)
    logger.info("RBM Training Completed.")

    # Evaluate on Test Set
    reconstructed_test = rbm.reconstruct(x_test)
    test_reconstruction_error = np.mean((x_test - reconstructed_test) ** 2)
    logger.info(f"Test Reconstruction Error: {test_reconstruction_error:.4f}")

    # Ensure directory for final model
    final_model_dir = os.path.dirname(MODEL_SAVE_PATH)
    if final_model_dir and not os.path.exists(final_model_dir):
        ensure_directory(final_model_dir)

    # Save the final model
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump({
            'weights': rbm.get_weights(),
            'visible_bias': rbm.get_visible_bias(),
            'hidden_bias': rbm.get_hidden_bias()
        }, f)
    logger.info(f"Final model saved to '{MODEL_SAVE_PATH}'.")

if __name__ == "__main__":
    main()
