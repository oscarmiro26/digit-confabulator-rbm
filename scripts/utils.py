# scripts/utils.py

import numpy as np

def binarize_data(data, threshold=127):
    """
    Binarize the data based on a threshold.

    Parameters:
    - data (np.ndarray): Array of shape (num_samples, 784) representing flattened MNIST images.
    - threshold (int, optional): Pixel value threshold for binarization. Defaults to 127.

    Returns:
    - np.ndarray: Binarized data with values 0 or 1.
    """
    return (data > threshold).astype(np.float32)

def normalize_data(data):
    """
    Normalize the data to the range [0, 1].

    Parameters:
    - data (np.ndarray): Array of shape (num_samples, 784) representing flattened MNIST images.

    Returns:
    - np.ndarray: Normalized data with values between 0 and 1.
    """
    return data.astype(np.float32) / 255.0

def preprocess_data(data, is_binary=True):
    """
    Preprocess the data based on the visibility type.

    Parameters:
    - data (list of lists or np.ndarray): Raw image data as lists of pixel values.
    - is_binary (bool, optional): If True, binarize the data; otherwise, normalize it. Defaults to True.

    Returns:
    - np.ndarray: Preprocessed data ready for RBM training.
    
    Raises:
    - ValueError: If `is_binary` is not a boolean.
    """
    if not isinstance(is_binary, bool):
        raise ValueError("is_binary must be a boolean value.")
    
    data = np.array(data).astype(np.float32)
    if is_binary:
        data = binarize_data(data)
    else:
        data = normalize_data(data)
    return data
