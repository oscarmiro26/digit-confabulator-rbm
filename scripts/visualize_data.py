# scripts/visualize_data.py

import numpy as np
import os
from os.path import join, dirname, abspath
import matplotlib.pyplot as plt
import random
from load_data import MnistDataloader

def setup_file_paths():
    """
    Set up file paths for the MNIST dataset relative to this script.

    Returns:
    - tuple: (training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    """
    # Get the directory where this script is located
    script_dir = dirname(abspath(__file__))

    # Navigate to the project root by going up one level
    project_root = dirname(script_dir)

    # Define the path to the 'data/raw/' directory
    base_path = join(project_root, 'data', 'raw')

    # Define file paths
    training_images_filepath = join(base_path, 'train-images.idx3-ubyte')
    training_labels_filepath = join(base_path, 'train-labels.idx1-ubyte')
    test_images_filepath = join(base_path, 't10k-images.idx3-ubyte')
    test_labels_filepath = join(base_path, 't10k-labels.idx1-ubyte')

    return training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath

def load_mnist_data(is_binary=True):
    """
    Load the MNIST dataset using MnistDataloader with specified preprocessing.

    Parameters:
    - is_binary (bool, optional): If True, binarize the data; otherwise, normalize it. Defaults to True.

    Returns:
    - tuple: ((x_train, y_train), (x_test, y_test))
    """
    training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath = setup_file_paths()

    mnist_dataloader = MnistDataloader(
        training_images_filepath, 
        training_labels_filepath, 
        test_images_filepath, 
        test_labels_filepath,
        is_binary=is_binary  # Updated parameter name
    )

    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    return (x_train, y_train), (x_test, y_test)

def count_classes(labels):
    """
    Count the number of observations for each class.

    Parameters:
    - labels (np.ndarray): Array of labels.

    Returns:
    - dict: Dictionary with class labels as keys and counts as values.
    """
    unique, counts = np.unique(labels, return_counts=True)
    class_counts = dict(zip(unique, counts))
    return class_counts

def visualize_one_per_class(data, labels, title_prefix=''):
    """
    Visualize one example for each class.

    Parameters:
    - data (np.ndarray): Array of image data.
    - labels (np.ndarray): Array of labels corresponding to the data.
    - title_prefix (str, optional): Prefix for the subplot titles. Defaults to ''.
    """
    classes = np.unique(labels)
    plt.figure(figsize=(10, 4))
    for cls in classes:
        # Find the first occurrence of the class
        index = np.where(labels == cls)[0][0]
        image = np.array(data[index]).reshape(28, 28)
        plt.subplot(2, 5, cls + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f"{title_prefix} Class {cls}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    # Choose preprocessing type
    is_binary = False

    # Load data
    (x_train, y_train), (x_test, y_test) = load_mnist_data(is_binary=is_binary)

    # Count classes
    train_class_counts = count_classes(y_train)
    test_class_counts = count_classes(y_test)

    # Print class counts
    print("Training Set Class Counts:")
    for cls in sorted(train_class_counts.keys()):
        print(f"Digit {cls}: {train_class_counts[cls]} samples")

    print("\nTest Set Class Counts:")
    for cls in sorted(test_class_counts.keys()):
        print(f"Digit {cls}: {test_class_counts[cls]} samples")

    # Print sizes of data splits
    print(f"\nTraining Set Size: {len(x_train)} samples")
    print(f"Test Set Size: {len(x_test)} samples")

    # Visualize one example per class from training set
    print("\nVisualizing one example per class from the Training Set:")
    visualize_one_per_class(x_train, y_train, title_prefix='Training')

    # Visualize one example per class from test set
    print("Visualizing one example per class from the Test Set:")
    visualize_one_per_class(x_test, y_test, title_prefix='Test')

if __name__ == "__main__":
    main()
