# scripts/evaluate.py

import sys
import os
import numpy as np
import pickle
import argparse

# Define project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.rbm import RBM
from load_data import MnistDataloader
import logging
import matplotlib.pyplot as plt
from utils import setup_logging, ensure_directory, load_model

def parse_arguments():
    """
    Parse command-line arguments for evaluation settings.
    
    Returns:
    - argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Evaluate Trained RBM on MNIST Dataset')
    parser.add_argument('--model_path', type=str, default='saved_models/rbm_final_model.pkl', help='Path to the trained RBM model file (e.g., rbm_final_model.pkl)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--is_binary', type=lambda x: (str(x).lower() == 'true'), default=True, help='Use binary visible units if True, else real-valued')
    parser.add_argument('--save_reconstructions', type=lambda x: (str(x).lower() == 'true'), default=True, help='Whether to save reconstruction images')
    parser.add_argument('--num_reconstructions', type=int, default=10, help='Number of reconstruction samples to save')
    parser.add_argument('--save_dir', type=str, default='generated', help='Directory to save evaluation outputs')
    
    # Modify this line to handle interactive environments
    return parser.parse_args(args=[])

def load_test_data(is_binary):
    """
    Load the MNIST test dataset.

    Parameters:
    - is_binary (bool): Whether to preprocess data as binary.

    Returns:
    - tuple: (x_test, y_test)
    """
    # Define file paths relative to project_root
    base_path = os.path.join(project_root, 'data', 'raw')
    test_images_filepath = os.path.join(base_path, 't10k-images.idx3-ubyte')
    test_labels_filepath = os.path.join(base_path, 't10k-labels.idx1-ubyte')

    # Initialize data loader
    data_loader = MnistDataloader(
        training_images_filepath=None,
        training_labels_filepath=None,
        test_images_filepath=test_images_filepath,
        test_labels_filepath=test_labels_filepath,
        is_binary=is_binary
    )

    # Load data
    _, _, test_data = data_loader.load_data()
    if test_data is None or test_data[0].size == 0:
        raise ValueError("Test data could not be loaded.")
    
    x_test, y_test = test_data
    return x_test, y_test

def visualize_reconstructions(original, reconstructed, save_path, num_images=10):
    """
    Visualize and save original and reconstructed images side by side.
    
    Parameters:
    - original (np.ndarray): Original images.
    - reconstructed (np.ndarray): Reconstructed images.
    - save_path (str): Path to save the visualization image.
    - num_images (int): Number of images to visualize.
    """
    num_images = min(num_images, original.shape[0])
    plt.figure(figsize=(num_images * 2, 4))
    for i in range(num_images):
        # Original Image
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == num_images // 2:
            ax.set_title('Original Images', fontsize=16)
        
        # Reconstructed Image
        ax = plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == num_images // 2:
            ax.set_title('Reconstructed Images', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_rbm(rbm, x_test, y_test, args, logger):
    """
    Perform RBM evaluation.

    Parameters:
    - rbm (RBM): Trained RBM instance.
    - x_test (np.ndarray): Test data.
    - y_test (np.ndarray): Test labels.
    - args (argparse.Namespace): Parsed arguments.
    - logger (logging.Logger): Logger instance.
    """
    num_samples = x_test.shape[0]
    batch_size = args.batch_size
    num_batches = num_samples // batch_size
    total_error = 0.0

    for batch_idx in range(num_batches):
        batch = x_test[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        reconstructed = rbm.reconstruct(batch)
        batch_error = np.mean((batch - reconstructed) ** 2)
        total_error += batch_error

    # Handle remaining samples
    if num_samples % batch_size != 0:
        batch = x_test[num_batches * batch_size :]
        reconstructed = rbm.reconstruct(batch)
        batch_error = np.mean((batch - reconstructed) ** 2)
        total_error += batch_error
        num_batches += 1

    avg_test_error = total_error / num_batches
    logger.info(f"Average Test Reconstruction Error (MSE): {avg_test_error:.4f}")

    # Optionally, visualize and save some reconstructions
    if args.save_reconstructions:
        # Select random samples from test set
        num_reconstructions = min(args.num_reconstructions, x_test.shape[0])
        indices = np.random.choice(num_samples, num_reconstructions, replace=False)
        selected_original = x_test[indices]
        reconstructed_selected = rbm.reconstruct(selected_original)

        # Define save path relative to project_root
        visualization_path = os.path.join(project_root, args.save_dir, 'reconstructions.png')
        visualize_reconstructions(selected_original, reconstructed_selected, visualization_path, num_images=num_reconstructions)
        logger.info(f"Reconstruction images saved to '{visualization_path}'.")

def main():
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    setup_logging(log_file='evaluation.log', project_root=project_root)
    logger = logging.getLogger(__name__)

    logger.info("RBM Evaluation Started.")
    logger.info(f"Model Path: {args.model_path}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Visible Units Type: {'Binary' if args.is_binary else 'Real-valued'}")
    logger.info(f"Save Reconstructions: {args.save_reconstructions}")
    logger.info(f"Number of Reconstructions to Save: {args.num_reconstructions}")
    logger.info(f"Save Directory: {args.save_dir}")

    # Ensure save directory exists relative to project_root
    save_dir_path = os.path.join(project_root, args.save_dir)
    ensure_directory(save_dir_path)

    # Load test data
    x_test, y_test = load_test_data(args.is_binary)
    logger.info(f"Test Set Size: {x_test.shape[0]}")

    # Initialize RBM
    n_visible = x_test.shape[1]
    # Temporary n_hidden; will be overwritten by loaded model
    rbm = RBM(
        n_visible=n_visible,
        n_hidden=64,  # Placeholder; actual value from model
        visible_type='binary' if args.is_binary else 'real',
        batch_size=args.batch_size
    )

    # Load model parameters
    load_model(args.model_path, rbm, project_root)
    n_hidden = rbm.get_weights().shape[1]  # Update n_hidden based on loaded model
    logger.info(f"RBM initialized with {n_visible} visible units and {n_hidden} hidden units.")

    # Perform evaluation
    evaluate_rbm(
        rbm=rbm,
        x_test=x_test,
        y_test=y_test,
        args=args,
        logger=logger
    )

    logger.info("RBM Evaluation Completed.")

if __name__ == "__main__":
    main()
