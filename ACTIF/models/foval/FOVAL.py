import numpy as np
import torch.nn as nn


class FOVAL(nn.Module):
    def __init__(self, input_size=38, embed_dim=150, dropout_rate=0.5, output_size=1, fc1_dim=32):
        super(FOVAL, self).__init__()
        self.modelName = "Foval"
        self.hidden_layer_size = embed_dim
        self.input_size = input_size

        # Linear layer to transform input features if needed
        self.input_linear = nn.Linear(in_features=self.input_size, out_features=self.input_size)

        self.dropout = nn.Dropout(p=dropout_rate)

        self.lstm = nn.LSTM(input_size=self.input_size, num_layers=1, batch_first=True,
                            hidden_size=self.hidden_layer_size)
        self.layernorm = nn.LayerNorm(self.hidden_layer_size)
        self.batchnorm = nn.BatchNorm1d(self.hidden_layer_size)

        # Additional fully connected layers
        self.fc1 = nn.Linear(self.hidden_layer_size, np.floor_divide(fc1_dim, 4))  # First additional FC layer
        self.fc5 = nn.Linear(np.floor_divide(fc1_dim, 4), output_size)  # Final FC layer for output
        self.activation = nn.ELU()
        self.print_model_parameter_size()

    def print_model_parameter_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("CURRENT MODEL IS: ", self.modelName)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")

    def forward(self, input_seq, return_intermediates=False):
        intermediates = {'Input': input_seq}
        in_linear = self.input_linear(input_seq)

        lstm_out, _ = self.lstm(input_seq)
        # Permute and apply batch normalization
        lstm_out_1 = lstm_out.permute(0, 2, 1)  # Change to (batch_size, num_features, seq_length)
        lstm_out_2 = self.batchnorm(lstm_out_1)
        lstm_out_3 = lstm_out_2.permute(0, 2, 1)  # Change back to (batch_size, seq_length, num_features)

        lstm_out_max_timestep, _ = lstm_out_3.max(dim=1)  # 75 start
        lstm_dropout = self.dropout(lstm_out_max_timestep)
        fc1_out = self.fc1(lstm_dropout)
        fc1_elu_out = self.activation(fc1_out)
        predictions = self.fc5(fc1_elu_out)

        if return_intermediates:
            intermediates['LSTM'] = lstm_out
            intermediates['Max Timestep'] = lstm_out_max_timestep
            intermediates['Output FC 1'] = fc1_out
            intermediates['ELU'] = fc1_elu_out
            intermediates['Output'] = predictions
            return predictions, intermediates
        else:
            return predictions
