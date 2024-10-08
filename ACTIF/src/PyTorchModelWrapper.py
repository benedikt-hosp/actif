import pandas as pd
import torch


class PyTorchModelWrapper:
    def __init__(self, model, input_shape):
        self.model = model
        self.input_shape = input_shape  # Expected shape (e.g., [sequence_length, num_features])

    def __call__(self, X):
        # Convert input to NumPy array if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Convert to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Ensure the input is reshaped to match the model's expected input shape (e.g., [batch_size, 10, num_features])
        if X_tensor.shape[1:] != self.input_shape:
            # Handle reshaping if needed
            expected_size = torch.Size([-1] + list(self.input_shape))
            if X_tensor.numel() % expected_size.numel() != 0:
                raise ValueError(f"Input shape {X_tensor.shape} cannot be reshaped to {expected_size}")
            X_tensor = X_tensor.view(-1, *self.input_shape)

        # Move the input to the same device as the model
        device = next(self.model.parameters()).device
        X_tensor = X_tensor.to(device)

        # Forward pass through the model
        with torch.no_grad():
            model_output = self.model(X_tensor, return_intermediates=False)

        # Return the model output as a NumPy array for compatibility with SHAP
        return model_output.cpu().numpy()


#
# class PyTorchModelWrapper:
#     def __init__(self, model, input_shape):
#         self.model = model
#         self.input_shape = input_shape  # Original input shape (without batch size)
#
#     def __call__(self, X):
#         if isinstance(X, pd.DataFrame):
#             X = X.values
#
#         # Convert to tensor
#         X_tensor = torch.tensor(X, dtype=torch.float32)
#
#         # Reshape the tensor to its original shape
#         X_tensor = X_tensor.view(-1, *self.input_shape)
#
#         # Move to the same device as the model
#         device = next(self.model.parameters()).device
#         X_tensor = X_tensor.to(device)
#
#         # Apply the model
#         with torch.no_grad():
#             model_output = self.model(X_tensor, return_intermediates=False)
#
#         return model_output.cpu().numpy()
#         # return model_output


class PyTorchModelWrapper2:
    def __init__(self, model, input_shape):
        self.model = model
        self.input_shape = input_shape  # Tuple like (sequence_length, num_features)

    def __call__(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Convert to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Log the shape of the incoming data
        print("Original X_tensor shape:", X_tensor.shape)

        # Reshape the tensor to its original shape
        try:
            X_tensor = X_tensor.view(-1, *self.input_shape)
            print("Reshaped X_tensor shape:", X_tensor.shape)
        except RuntimeError as e:
            print(f"Error reshaping X_tensor: {e}")
            # Handle error, e.g., by adjusting input_shape or handling variable sequence lengths
            raise

        # Move to the same device as the model
        device = next(self.model.parameters()).device
        X_tensor = X_tensor.to(device)

        # Apply the model
        with torch.no_grad():
            model_output = self.model(X_tensor, return_intermediates=False)

        return model_output.cpu().numpy()
