import numpy as np
import torch.nn as nn

import torch.nn as nn

#
# class SimpleLSTM_V3(nn.Module):
#     def __init__(self, input_size=38, embed_dim=150, dropout_rate=0.5, output_size=1, fc1_dim=32):
#         super(SimpleLSTM_V3, self).__init__()
#         self.hidden_layer_size = embed_dim
#         self.input_size = input_size
#
#         # Linear layer to transform input features if needed
#         self.input_linear = nn.Linear(in_features=self.input_size, out_features=self.input_size)
#
#         # LSTM layer
#         self.lstm = nn.LSTM(input_size=self.input_size, num_layers=1, batch_first=True,
#                             hidden_size=self.hidden_layer_size)
#         self.layernorm = nn.LayerNorm(self.hidden_layer_size)
#         self.batchnorm = nn.BatchNorm1d(self.hidden_layer_size)
#
#         # Additional fully connected layers
#         self.fc1 = nn.Linear(self.hidden_layer_size, fc1_dim)  # First additional FC layer
#         self.fc5 = nn.Linear(fc1_dim, output_size)  # Final FC layer for output
#         self.activation = nn.ELU()
#
#         # Dropout layer
#         self.dropout = nn.Dropout(p=dropout_rate)
#
#     def forward(self, input_seq, return_intermediates=True):
#
#         intermediates = {'input_seq': input_seq}
#
#         # Capture the input activations after applying the linear transformation
#         input_activations = self.input_linear(input_seq)
#         intermediates['Input_activations'] = input_activations
#
#         # Save the weight matrix of the first linear layer
#         intermediates['Input_Weights'] = self.input_linear.weight.data.cpu().numpy()
#
#         # Pass the activations through the LSTM layer
#         lstm_out, _ = self.lstm(input_activations)
#         intermediates['LSTM_Out'] = lstm_out
#
#         # Save the weight matrices for the LSTM layer
#         intermediates['LSTM_Weights_IH'] = self.lstm.weight_ih_l0.data.cpu().numpy()  # Input-hidden weights
#         intermediates['LSTM_Weights_HH'] = self.lstm.weight_hh_l0.data.cpu().numpy()  # Hidden-hidden weights
#
#         # Apply non-linear activation (ELU)
#         lstm_activated = self.activation(lstm_out)
#         intermediates['LSTM_ELU'] = lstm_activated
#
#         # Permute and apply batch normalization
#         lstm_out_1 = lstm_activated.permute(0, 2, 1)  # Change to (batch_size, num_features, seq_length)
#         lstm_out_2 = self.batchnorm(lstm_out_1)
#         lstm_out_3 = lstm_out_2.permute(0, 2, 1)  # Change back to (batch_size, seq_length, num_features)
#
#         # Max pooling over the time dimension
#         lstm_out_max_timestep, _ = lstm_out_3.max(dim=1)  # Max-pooling over time
#         intermediates['Max_Timestep'] = lstm_out_max_timestep
#
#         lstm_dropout = self.dropout(lstm_out_max_timestep)
#         fc1_out = self.fc1(lstm_dropout)
#         intermediates['FC1_Out'] = fc1_out
#
#         # Save the weight matrix for the first fully connected layer
#         intermediates['FC1_Weights'] = self.fc1.weight.data.cpu().numpy()
#
#         fc1_elu_out = self.activation(fc1_out)
#         intermediates['FC1_ELU_Out'] = fc1_elu_out
#
#         predictions = self.fc5(fc1_elu_out)
#         intermediates['Output'] = predictions
#
#         # Save the weight matrix for the final fully connected layer
#         intermediates['FC5_Weights'] = self.fc5.weight.data.cpu().numpy()
#
#         if return_intermediates:
#             return predictions, intermediates
#         else:
#             return predictions

class SimpleLSTM_V2(nn.Module):
    def __init__(self, input_size=4, embed_dim=150, dropout_rate=0.5, output_size=1, fc1_dim=32):
        super(SimpleLSTM_V2, self).__init__()
        self.hidden_layer_size = embed_dim

        # Linear layer to transform input features if needed
        self.input_linear = nn.Linear(in_features=input_size, out_features=input_size)

        self.dropout = nn.Dropout(p=dropout_rate)

        self.lstm = nn.LSTM(input_size=input_size, num_layers=1, batch_first=True, hidden_size=self.hidden_layer_size)
        self.layernorm = nn.LayerNorm(self.hidden_layer_size)
        self.batchnorm = nn.BatchNorm1d(self.hidden_layer_size)

        # Additional fully connected layers
        self.fc1 = nn.Linear(self.hidden_layer_size, np.floor_divide(fc1_dim, 4))  # First additional FC layer
        self.fc5 = nn.Linear(np.floor_divide(fc1_dim, 4), output_size)  # Final FC layer for output
        self.activation = nn.ELU()

    def forward(self, input_seq, return_intermediates=False):
        intermediates = {'Input': input_seq}

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

