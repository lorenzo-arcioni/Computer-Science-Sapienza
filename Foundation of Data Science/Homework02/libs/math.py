import numpy as np


def sigmoid(x):
    """
    Function to compute the sigmoid of a given input x.

    Args:
        x: it's the input data matrix.

    Returns:
        g: The sigmoid of the input x
    """
    ##############################
    ###     YOUR CODE HERE     ###
    ##############################    
    return 1 / (1 + np.exp(- np.clip(x, -20, 20)))

def softmax(y):
    """
    Function to compute associated probability for each sample and each class.

    Args:
        y: the predicted 

    Returns:
        softmax_scores: it's the matrix containing probability for each sample and each class. The shape is (N, K)
    """
    
    # Subtract the maximum value in each row to prevent numerical overflow and compute the exponentials
    exp_y = np.exp(y - np.max(y, axis=1, keepdims=True))
    
    # Normalize each row to get probabilities
    softmax_scores = exp_y / np.sum(exp_y, axis=1, keepdims=True)
    
    return softmax_scores

