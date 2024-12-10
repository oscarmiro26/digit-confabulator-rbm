# scripts/visualize_data.py

import numpy as np
import os
import sys
from os.path import join, dirname, abspath

# Define project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import matplotlib.pyplot as plt
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
    test_labels_filepath = join(base_path, 'train-labels.idx1-ubyte')

    return training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath


def load_mnist_data(is_binary=True, validation_split=0.1):
    """
    Load the MNIST dataset using MnistDataloader with specified preprocessing and validation split.

    Parameters:
    - is_binary (bool, optional): If True, binarize the data; otherwise, normalize it. Defaults to True.
    - validation_split (float, optional): Fraction of training data to use for validation. Defaults to 0.1.

    Returns:
    - tuple: ((x_train, y_train), (x_val, y_val), (x_test, y_test))
    """
    training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath = setup_file_paths()

    mnist_dataloader = MnistDataloader(
        training_images_filepath=training_images_filepath,
        training_labels_filepath=training_labels_filepath,
        test_images_filepath=test_images_filepath,
        test_labels_filepath=test_labels_filepath,
        is_binary=is_binary,
        validation_split=validation_split
    )

    return mnist_dataloader.load_data()


def count_classes(labels):
    """
    Count the number of observations for each class.

    Parameters:
    - labels (np.ndarray): Array of labels.

    Returns:
    - dict: Dictionary with class labels as keys and counts as values.
    """
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))


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
    # Configuration
    is_binary = False
    validation_split = 0.1

    # Load data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist_data(is_binary=is_binary, validation_split=validation_split)

    # Count classes
    train_class_counts = count_classes(y_train)
    val_class_counts = count_classes(y_val)
    test_class_counts = count_classes(y_test)

    # Print class counts
    print("Training Set Class Counts:")
    for cls in sorted(train_class_counts.keys()):
        print(f"Digit {cls}: {train_class_counts[cls]} samples")

    print("\nValidation Set Class Counts:")
    for cls in sorted(val_class_counts.keys()):
        print(f"Digit {cls}: {val_class_counts[cls]} samples")

    print("\nTest Set Class Counts:")
    for cls in sorted(test_class_counts.keys()):
        print(f"Digit {cls}: {test_class_counts[cls]} samples")

    # Print sizes of data splits
    print(f"\nTraining Set Size: {len(x_train)} samples")
    print(f"Validation Set Size: {len(x_val)} samples")
    print(f"Test Set Size: {len(x_test)} samples")

    # Visualize one example per class from training set
    print("\nVisualizing one example per class from the Training Set:")
    visualize_one_per_class(x_train, y_train, title_prefix='Training')

    # Visualize one example per class from validation set
    print("Visualizing one example per class from the Validation Set:")
    visualize_one_per_class(x_val, y_val, title_prefix='Validation')

    # Visualize one example per class from test set
    print("Visualizing one example per class from the Test Set:")
    visualize_one_per_class(x_test, y_test, title_prefix='Test')


if __name__ == "__main__":
    main()
