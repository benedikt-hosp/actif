import pandas as pd
import torch

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
#             model_output = self.model(X_tensor)
#
#         return model_output.cpu().numpy()

class PyTorchModelWrapper:
    def __init__(self, model, expected_size):
        self.model = model
        self.expected_size = expected_size  # Example: (10, 38) representing (time_steps, num_features)

    def __call__(self, data):
        # Convert input data to a tensor and move it to the appropriate device
        X_tensor = torch.tensor(data).float().to(next(self.model.parameters()).device)

        # Calculate the actual batch size dynamically from the input data
        batch_size = X_tensor.size(0)

        # Reshape the input to match the model's expected input size
        # Note: `expected_size` is (time_steps, num_features), but the batch size is dynamic
        time_steps, num_features = self.expected_size
        try:
            # Reshape using the dynamic batch size and predefined time_steps, num_features
            X_tensor = X_tensor.view(batch_size, time_steps, num_features)
        except Exception as e:
            raise ValueError(
                f"Input shape {X_tensor.shape} cannot be reshaped to ({batch_size}, {time_steps}, {num_features})")

        # Perform forward pass through the model
        return self.model(X_tensor).detach().cpu().numpy()
