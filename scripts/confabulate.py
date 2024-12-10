# scripts/confabulate.py

import sys
import os
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt

# Define project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


from models.rbm import RBM
import logging
from utils import setup_logging, ensure_directory
 

def parse_arguments():
    """
    Parse command-line arguments for confabulation settings.

    Returns:
    - argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Confabulate Images Using Trained RBM')
    parser.add_argument('--model_path', type=str, default='saved_models/rbm_final_model.pkl',
                        help='Path to the trained RBM model file (e.g., rbm_final_model.pkl)')
    parser.add_argument('--num_images', type=int, default=10,
                        help='Number of images to generate')
    parser.add_argument('--num_gibbs_steps', type=int, default=1000,
                        help='Number of Gibbs sampling steps for each image')
    parser.add_argument('--k', type=int, default=1,
                        help='Number of Gibbs sampling steps in Contrastive Divergence')
    parser.add_argument('--is_binary', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='Use binary visible units if True, else real-valued')
    parser.add_argument('--save_dir', type=str, default='generated',
                        help='Directory to save generated images')
    return parser.parse_args(args=[])


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


def get_next_file_name(save_dir):
    """
    Get the next unique file name based on existing files in the directory.

    Parameters:
    - save_dir (str): Path to the directory where images are saved.

    Returns:
    - str: The next file name (e.g., 'generated_digits_X.png').
    """
    files = [f for f in os.listdir(save_dir) if f.startswith("generated_digits_") and f.endswith(".png")]
    next_index = len(files)  # Use the count of existing files as the next index
    return f"generated_digits_{next_index}.png"


def visualize_generated_images_with_labels(generated_images, save_path, num_images=10):
    """
    Visualize and save generated images in a single grid file, with labels below each image.

    Parameters:
    - generated_images (np.ndarray): Array of generated images.
    - save_path (str): Path to save the visualization image.
    - num_images (int): Number of images to display in the grid.
    """
    grid_size = int(np.ceil(np.sqrt(num_images)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 3, grid_size * 3))
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        if idx < num_images:
            ax.imshow(generated_images[idx].reshape(28, 28), cmap='gray')
            ax.axis('off')
            # Add label below the image
            ax.set_title(f"Image {idx + 1}", fontsize=10, pad=10)
        else:
            # Hide unused subplots
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def confabulate(rbm, num_images, num_gibbs_steps, save_dir, project_root, logger):
    """
    Generate images using Gibbs sampling with the trained RBM.

    Parameters:
    - rbm (RBM): Trained RBM instance.
    - num_images (int): Number of images to generate.
    - num_gibbs_steps (int): Total number of Gibbs sampling steps per image.
    - save_dir (str): Directory to save generated images.
    - project_root (str): Absolute path to the project root directory.
    - logger (logging.Logger): Logger instance.
    """
    # Ensure save directory exists
    absolute_save_dir = os.path.join(project_root, save_dir)
    ensure_directory(absolute_save_dir)
    
    logger.info("Starting image generation...")
    generated_images = []

    for img_idx in range(num_images):
        # Initialize visible units
        if rbm.visible_type == 'binary':
            visible = np.random.binomial(1, 0.5, size=(1, rbm.n_visible)).astype(np.float32)
        else:
            visible = np.random.normal(0, 1, size=(1, rbm.n_visible)).astype(np.float32)

        logger.info(f"Generating image {img_idx + 1}/{num_images}...")

        # Gibbs sampling
        for _ in range(num_gibbs_steps):
            # Sample hidden units given visible units
            hidden_probs, hidden_states = rbm.compute_hidden(visible)
            # Sample visible units given hidden units
            visible_probs, visible_states = rbm.compute_visible(hidden_states)
            visible = visible_states  # Update visible state

        generated_images.append(visible.flatten())

    # Convert list to NumPy array
    generated_images = np.array(generated_images)

    # Define save path
    save_file_name = get_next_file_name(absolute_save_dir)
    visualization_path = os.path.join(absolute_save_dir, save_file_name)
    visualize_generated_images_with_labels(generated_images, visualization_path, num_images=num_images)

    logger.info(f"Generated images saved to '{visualization_path}'.")


def main():
    # Parse arguments
    args = parse_arguments()

    # Define project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Define log directory
    log_dir = os.path.join(project_root, 'logs')
    ensure_directory(log_dir)

    # Setup logging
    log_file_path = os.path.join(log_dir, 'confabulation.log')
    setup_logging(log_file=log_file_path, project_root=project_root)
    logger = logging.getLogger(__name__)

    logger.info("RBM Confabulation Started.")
    logger.info(f"Model Path: {args.model_path}")
    logger.info(f"Number of Images to Generate: {args.num_images}")
    logger.info(f"Number of Gibbs Steps: {args.num_gibbs_steps}")
    logger.info(f"Number of Gibbs Sampling Steps (k): {args.k}")
    logger.info(f"Visible Units Type: {'Binary' if args.is_binary else 'Real-valued'}")
    logger.info(f"Save Directory: {args.save_dir}")

    # Initialize RBM
    rbm = RBM(
        n_visible=784,  # MNIST images are 28x28
        n_hidden=64,     # Placeholder; actual value from model
        learning_rate=0.1,  # Placeholder; not used in confabulation
        momentum=0.5,       # Placeholder; not used in confabulation
        weight_decay=0.0002, # Placeholder; not used in confabulation
        use_pcd=False,      # Placeholder; not used in confabulation
        k=args.k,           # Correctly defined now
        visible_type='binary' if args.is_binary else 'real',
        batch_size=args.num_images  # Batch size is the number of images to generate
    )

    # Load model
    try:
        load_model(args.model_path, rbm, project_root)
        logger.info("RBM model loaded successfully.")
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Perform confabulation
    confabulate(
        rbm=rbm,
        num_images=args.num_images,
        num_gibbs_steps=args.num_gibbs_steps,
        save_dir=args.save_dir,
        project_root=project_root,
        logger=logger
    )

    logger.info("RBM Confabulation Completed.")


if __name__ == "__main__":
    main()
