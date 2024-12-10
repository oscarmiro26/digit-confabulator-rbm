# scripts/load_data.py

import numpy as np
import struct
from array import array
from os.path import join, dirname, abspath
import os
from scripts.data_preprocessing import preprocess_data  # Ensure this path is correct

class MnistDataloader:
    """
    A class to load and preprocess the MNIST dataset.

    Attributes:
    - training_images_filepath (str or None): Path to the training images file.
    - training_labels_filepath (str or None): Path to the training labels file.
    - test_images_filepath (str or None): Path to the test images file.
    - test_labels_filepath (str or None): Path to the test labels file.
    - is_binary (bool): Determines if the data should be binarized or normalized.
    - validation_split (float): Fraction of training data to use for validation (between 0 and 1).
    """
    
    def __init__(self, training_images_filepath=None, training_labels_filepath=None,
                 test_images_filepath=None, test_labels_filepath=None, 
                 is_binary=True, validation_split=0.1):
        """
        Initialize the MnistDataloader with file paths, preprocessing type, and validation split.

        Parameters:
        - training_images_filepath (str or None): Path to the training images file.
        - training_labels_filepath (str or None): Path to the training labels file.
        - test_images_filepath (str or None): Path to the test images file.
        - test_labels_filepath (str or None): Path to the test labels file.
        - is_binary (bool, optional): If True, binarize the data; otherwise, normalize it. Defaults to True.
        - validation_split (float, optional): Fraction of training data to use for validation. Must be between 0 and 1. Defaults to 0.1.
        
        Raises:
        - ValueError: If validation_split is not between 0 and 1.
        """
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
        self.is_binary = is_binary
        self.validation_split = validation_split

        # Validate validation_split
        if not (0.0 <= self.validation_split < 1.0):
            raise ValueError("validation_split must be a float between 0 (inclusive) and 1 (exclusive).")
    
    def read_images_labels(self, images_filepath, labels_filepath):
        """
        Read images and labels from the specified file paths.

        Parameters:
        - images_filepath (str): Path to the images file.
        - labels_filepath (str): Path to the labels file.

        Returns:
        - tuple: (images, labels) where images is a NumPy array of images and labels is a NumPy array of labels.

        Raises:
        - ValueError: If magic numbers do not match expected values.
        """
        if not images_filepath or not labels_filepath:
            return np.array([]), np.array([])  # Return empty arrays if file paths are not provided

        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f'Magic number mismatch in labels file: expected 2049, got {magic}')
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f'Magic number mismatch in images file: expected 2051, got {magic}')
            image_data = array("B", file.read())

        images = []
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            images.append(img)

        images = np.array(images)  # Convert list to NumPy array for efficient processing

        return images, np.array(labels)

    def load_data(self):
        """
        Load and preprocess the MNIST dataset.

        Returns:
        - tuple: ((x_train, y_train), (x_val, y_val), (x_test, y_test)) where
                 x_train and y_train are training data and labels,
                 x_val and y_val are validation data and labels,
                 x_test and y_test are test data and labels.
                 If training data is not provided, x_train and y_train will be empty arrays.
        """
        # Load training data if file paths are provided
        if self.training_images_filepath and self.training_labels_filepath:
            x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
            if x_train.size > 0 and y_train.size > 0:
                # Preprocess training data
                x_train = preprocess_data(x_train, is_binary=self.is_binary)

                # Shuffle the training data before splitting
                permutation = np.random.permutation(x_train.shape[0])
                x_train = x_train[permutation]
                y_train = y_train[permutation]

                # Split into training and validation sets based on validation_split
                split_idx = int((1 - self.validation_split) * x_train.shape[0])
                x_train_split, x_val_split = x_train[:split_idx], x_train[split_idx:]
                y_train_split, y_val_split = y_train[:split_idx], y_train[split_idx:]
            else:
                x_train_split, y_train_split = np.array([]), np.array([])
                x_val_split, y_val_split = np.array([]), np.array([])
        else:
            x_train_split, y_train_split = np.array([]), np.array([])
            x_val_split, y_val_split = np.array([]), np.array([])

        # Load test data if file paths are provided
        if self.test_images_filepath and self.test_labels_filepath:
            x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
            if x_test.size > 0 and y_test.size > 0:
                # Preprocess test data
                x_test = preprocess_data(x_test, is_binary=self.is_binary)
            else:
                x_test, y_test = np.array([]), np.array([])
        else:
            x_test, y_test = np.array([]), np.array([])

        return (x_train_split, y_train_split), (x_val_split, y_val_split), (x_test, y_test)
