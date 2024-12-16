# model/rbm.py

import numpy as np

class RBM:
    def __init__(self, n_visible, n_hidden, learning_rate=0.1, momentum=0.5, weight_decay=0.0002, 
                 use_pcd=True, k=1, is_binary=True, batch_size=64):
        """
        Initialize the RBM model parameters.

        Parameters:
        - n_visible: Number of visible units
        - n_hidden: Number of hidden units
        - learning_rate: Learning rate for weight updates
        - momentum: Momentum for weight updates
        - weight_decay: Weight decay (L2 regularization)
        - use_pcd: Whether to use Persistent Contrastive Divergence
        - k: Number of Gibbs sampling steps
        - is_binary: Type of visible units (true for binary, else real)
        - batch_size: Size of mini-batches for training
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_pcd = use_pcd
        self.k = k
        self.is_binary = is_binary
        self.batch_size = batch_size

        # Initialize weights using Xavier initialization
        limit = np.sqrt(6. / (self.n_visible + self.n_hidden))
        self.weights = np.random.uniform(-limit, limit, size=(self.n_visible, self.n_hidden))

        # Initialize biases to zero
        self.visible_bias = np.zeros(self.n_visible)
        self.hidden_bias = np.zeros(self.n_hidden)

        # Initialize velocity for momentum
        self.weights_velocity = np.zeros_like(self.weights)
        self.visible_bias_velocity = np.zeros_like(self.visible_bias)
        self.hidden_bias_velocity = np.zeros_like(self.hidden_bias)

        # Initialize persistent chain for PCD
        if self.use_pcd:
            self.persistent_visible = np.random.binomial(1, 0.5, size=(self.batch_size, self.n_visible)).astype(np.float32)

    def __str__(self):
        return (
            f"n_visible = {self.n_visible}\n" +
            f"n_hidden = {self.n_hidden}\n" +
            f"lr = {self.learning_rate}\n" +
            f"momentum = {self.momentum}\n" +
            f"weight_decay = {self.weight_decay}\n" +
            f"use_pcd = {self.use_pcd}\n" +
            f"k = {self.k}\n" +
            f"is_binary = {self.is_binary}\n" +
            f"batch_size = {self.batch_size}"
        )

    def sigmoid(self, x):
        """
        Compute the sigmoid function.

        Parameters:
        - x: Input array

        Returns:
        - Sigmoid of x
        """
        return 1.0 / (1 + np.exp(-x))

    def sample_prob(self, probs):
        """
        Sample binary states based on probabilities.

        Parameters:
        - probs: Probabilities for each unit

        Returns:
        - Binary samples (0 or 1)
        """
        return (probs > np.random.rand(*probs.shape)).astype(np.float32)

    def compute_hidden(self, visible):
        """
        Compute hidden unit activations given visible units.

        Parameters:
        - visible: Visible units (batch_size x n_visible)

        Returns:
        - hidden_probs: Probabilities of hidden units
        - hidden_states: Sampled hidden states
        """
        hidden_activations = np.dot(visible, self.weights) + self.hidden_bias
        hidden_probs = self.sigmoid(hidden_activations)
        hidden_states = self.sample_prob(hidden_probs)
        return hidden_probs, hidden_states

    def compute_visible(self, hidden):
        """
        Compute visible unit activations given hidden units.

        Parameters:
        - hidden: Hidden units (batch_size x n_hidden)

        Returns:
        - visible_probs: Probabilities of visible units
        - visible_states: Sampled visible states
        """
        visible_activations = np.dot(hidden, self.weights.T) + self.visible_bias
        if self.is_binary is True:
            visible_probs = self.sigmoid(visible_activations)
            visible_states = self.sample_prob(visible_probs)
        else:
            # For real-valued visible units, use Gaussian distribution
            # Assuming unit variance for simplicity
            visible_probs = visible_activations  # Mean of Gaussian
            visible_states = visible_activations + np.random.normal(0, 1, size=visible_activations.shape)
        return visible_probs, visible_states

    def contrastive_divergence(self, input_data):
        """
        Perform (Persistent) Contrastive Divergence (PCD-k) to update weights
        and biases.

        Parameters:
        - input_data: Input data (batch_size x n_visible)
        """
        # Positive phase
        pos_hidden_probs, _ = self.compute_hidden(input_data)
        pos_associations = np.dot(input_data.T, pos_hidden_probs)

        # Initialize negative samples
        if self.use_pcd:
            negative_visible = self.persistent_visible
        else:
            negative_visible = input_data.copy()

        # Gibbs sampling for k steps
        for _ in range(self.k):
            _, negative_hidden = self.compute_hidden(negative_visible)
            _, negative_visible = self.compute_visible(negative_hidden)

        if self.use_pcd:
            # Update persistent chain
            self.persistent_visible = negative_visible.copy()

        # Negative phase
        neg_hidden_probs, _ = self.compute_hidden(negative_visible)
        neg_associations = np.dot(negative_visible.T, neg_hidden_probs)

        # Update weights and biases
        # Compute gradients
        weight_update = (pos_associations - neg_associations) / input_data.shape[0]
        visible_bias_update = np.mean(input_data - negative_visible, axis=0)
        hidden_bias_update = np.mean(pos_hidden_probs - neg_hidden_probs, axis=0)

        # Apply weight decay (L2 regularization)
        weight_update -= self.weight_decay * self.weights

        # Update velocities
        self.weights_velocity = self.momentum * self.weights_velocity + self.learning_rate * weight_update
        self.visible_bias_velocity = self.momentum * self.visible_bias_velocity + self.learning_rate * visible_bias_update
        self.hidden_bias_velocity = self.momentum * self.hidden_bias_velocity + self.learning_rate * hidden_bias_update

        # Update parameters
        self.weights += self.weights_velocity
        self.visible_bias += self.visible_bias_velocity
        self.hidden_bias += self.hidden_bias_velocity

    def reconstruct(self, input_data):
        """
        Reconstruct visible units from input data.

        Parameters:
        - input_data: Input data (batch_size x n_visible)

        Returns:
        - reconstructed_visible_probs: Reconstructed visible probabilities
        """
        hidden_probs, hidden_states = self.compute_hidden(input_data)
        reconstructed_visible_probs, _ = self.compute_visible(hidden_probs)
        return reconstructed_visible_probs

    def get_weights(self):
        """
        Get the current weights of the RBM.

        Returns:
        - weights: Current weight matrix
        """
        return self.weights

    def get_visible_bias(self):
        """
        Get the current visible biases.

        Returns:
        - visible_bias: Current visible biases
        """
        return self.visible_bias

    def get_hidden_bias(self):
        """
        Get the current hidden biases.

        Returns:
        - hidden_bias: Current hidden biases
        """
        return self.hidden_bias

    def set_weights(self, weights):
        """
        Set the RBM weights.

        Parameters:
        - weights: New weight matrix
        """
        if weights.shape != self.weights.shape:
            raise ValueError(f"Weight shape mismatch. Expected {self.weights.shape}, got {weights.shape}")
        self.weights = weights

    def set_visible_bias(self, visible_bias):
        """
        Set the RBM visible biases.

        Parameters:
        - visible_bias: New visible biases
        """
        if visible_bias.shape != self.visible_bias.shape:
            raise ValueError(f"Visible bias shape mismatch. Expected {self.visible_bias.shape}, got {visible_bias.shape}")
        self.visible_bias = visible_bias

    def set_hidden_bias(self, hidden_bias):
        """
        Set the RBM hidden biases.

        Parameters:
        - hidden_bias: New hidden biases
        """
        if hidden_bias.shape != self.hidden_bias.shape:
            raise ValueError(f"Hidden bias shape mismatch. Expected {self.hidden_bias.shape}, got {hidden_bias.shape}")
        self.hidden_bias = hidden_bias
