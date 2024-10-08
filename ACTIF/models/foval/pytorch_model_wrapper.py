import pandas as pd
import torch


class PyTorchModelWrapper:
    def __init__(self, model, input_shape):
        self.model = model
        self.input_shape = input_shape  # Original input shape (without batch size)

    def __call__(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Convert to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Reshape the tensor to its original shape
        X_tensor = X_tensor.view(-1, *self.input_shape)

        # Move to the same device as the model
        device = next(self.model.parameters()).device
        X_tensor = X_tensor.to(device)

        # Apply the model
        with torch.no_grad():
            model_output = self.model(X_tensor)

        return model_output.cpu().numpy()

class PyTorchModelWrapper2:
    def __init__(self, model):
        self.model = model

    def __call__(self, X):
        # If X is a DataFrame, convert to NumPy array first
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Now X is guaranteed to be a NumPy array, convert it to a tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Move the tensor to the same device as the model
        device = next(self.model.parameters()).device
        X_tensor = X_tensor.to(device)

        # Apply the model and get the output
        with torch.no_grad():
            model_output = self.model(X_tensor)

        # Convert the output to a NumPy array and return
        return model_output.cpu().numpy()
