import numpy as np

def fit(model, x: np.array, y: np.array, x_val: np.array = None, y_val: np.array = None, lr: float = 0.5, num_steps: int = 500):
    """
    Function to fit the logistic regression model using gradient ascent.

    Args:
        model: the logistic regression model.
        x: it's the input data matrix.
        y: the label array.
        x_val: it's the input data matrix for validation.
        y_val: the label array for validation.
        lr: the learning rate.
        num_steps: the number of iterations.

    Returns:
        history: the values of the log likelihood during the process.
    """
    likelihood_history = np.zeros(num_steps)
    val_loss_history = np.zeros(num_steps)

    for it in range(num_steps):
        # Predict on the current parameters
        preds = model.predict(x)
        
        # Calculate log-likelihood and store it
        likelihood_history[it] = model.likelihood(preds, y)
        
        # Compute the gradient
        gradient = model.compute_gradient(x, y, preds)
        
        # Update the model parameters
        model.update_theta(gradient, lr)
        
        # Calculate validation loss if validation data is provided
        if x_val is not None and y_val is not None:
            val_preds = model.predict(x_val)
            val_loss_history[it] = - model.likelihood(val_preds, y_val)

    return likelihood_history, val_loss_history