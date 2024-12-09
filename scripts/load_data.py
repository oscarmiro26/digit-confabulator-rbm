# scripts/load_data.py

import numpy as np
import struct
from array import array
from os.path import join, dirname, abspath
import os
from utils import preprocess_data

class MnistDataloader:
    """
    A class to load and preprocess the MNIST dataset.

    Attributes:
    - training_images_filepath (str): Path to the training images file.
    - training_labels_filepath (str): Path to the training labels file.
    - test_images_filepath (str): Path to the test images file.
    - test_labels_filepath (str): Path to the test labels file.
    - is_binary (bool): Determines if the data should be binarized or normalized.
    """

    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath, is_binary=True):
        """
        Initialize the MnistDataloader with file paths and preprocessing type.

        Parameters:
        - training_images_filepath (str): Path to the training images file.
        - training_labels_filepath (str): Path to the training labels file.
        - test_images_filepath (str): Path to the test images file.
        - test_labels_filepath (str): Path to the test labels file.
        - is_binary (bool, optional): If True, binarize the data; otherwise, normalize it. Defaults to True.
        """
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
        self.is_binary = is_binary

    def read_images_labels(self, images_filepath, labels_filepath):
        """
        Read images and labels from the specified file paths.

        Parameters:
        - images_filepath (str): Path to the images file.
        - labels_filepath (str): Path to the labels file.

        Returns:
        - tuple: (images, labels) where images is a list of image pixel lists and labels is an array of labels.
        
        Raises:
        - ValueError: If magic numbers do not match expected values.
        """
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f'Magic number mismatch, expected 2049, got {magic}')
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f'Magic number mismatch, expected 2051, got {magic}')
            image_data = array("B", file.read())

        images = []
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            images.append(img.tolist())

        return images, labels

    def load_data(self):
        """
        Load and preprocess the MNIST dataset.

        Returns:
        - tuple: ((x_train, y_train), (x_test, y_test)) where
                 x_train and x_test are preprocessed image arrays,
                 y_train and y_test are label arrays.
        """
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)

        # Preprocess data
        x_train = preprocess_data(x_train, is_binary=self.is_binary)
        x_test = preprocess_data(x_test, is_binary=self.is_binary)

        y_train = np.array(y_train)
        y_test = np.array(y_test)

        return (x_train, y_train), (x_test, y_test)
