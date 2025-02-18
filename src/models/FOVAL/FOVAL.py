# import torch.nn as nn
# import numpy as np
#
#
# class MaxOverTimePooling(nn.Module):
#     def __init__(self):
#         super(MaxOverTimePooling, self).__init__()
#
#     def forward(self, x):
#         return x.max(dim=1)[0]  # Max über die Zeitdimension (seq_length)
#
#
# class Foval(nn.Module):
#     def __init__(self, device, feature_count):
#         super(Foval, self).__init__()
#
#         self.maxpool = None
#         self.device = device
#
#
#         # Hyperparameteres
#         self.hidden_layer_size = None
#         self.feature_count = feature_count
#         self.input_size = None
#         self.embed_dim = None
#         self.fc1_dim = None
#         self.fc5_dim = None
#         self.outputsize = 1
#         self.dropout_rate = None
#
#         # Layers
#         self.input_linear = None
#         self.lstm = None
#         self.layernorm = None
#         self.batchnorm = None
#         self.fc1 = None
#         self.fc5 = None
#         self.activation = None
#         self.dropout = None
#
#
#         self.modelName = "Foval"
#
#         self.do_inverse_transform = False
#         self.target_scaler = None
#
#         # self.to(device)
#         # Load Hyperparameteres from file
#
#     def initialize(self, input_size, hidden_layer_size, fc1_dim, dropout_rate):
#
#         self.hidden_layer_size = hidden_layer_size
#         self.input_size = input_size
#
#         # Linear layer to transform input features if needed
#         self.input_linear = nn.Linear(in_features=self.input_size, out_features=self.input_size)
#
#         self.dropout = nn.Dropout(p=dropout_rate)
#
#         self.lstm = nn.LSTM(input_size=self.input_size, num_layers=1, batch_first=True,
#                             hidden_size=self.hidden_layer_size)
#         self.layernorm = nn.LayerNorm(self.hidden_layer_size)
#         self.batchnorm = nn.BatchNorm1d(self.hidden_layer_size)
#
#         # 🔹 Explicitly Add MaxPool1d Layer
#         self.maxpool = nn.MaxPool1d(kernel_size=10)  # Pool over the entire sequence length
#
#         # Additional fully connected layers
#         self.fc1 = nn.Linear(self.hidden_layer_size, np.floor_divide(fc1_dim, 4))  # First additional FC layer
#         self.fc5 = nn.Linear(np.floor_divide(fc1_dim, 4), self.outputsize)  # Final FC layer for output
#         self.activation = nn.ELU()
#         # self.print_model_parameter_size()
#         self.to(self.device)
#
#
#         # # input_size = 34
#         # print("Hyperparameters of model: ", input_size, hidden_layer_size, fc1_dim, dropout_rate)
#         # # Linear layer to transform input features if needed
#         # self.input_linear = nn.Linear(in_features=input_size, out_features=input_size)
#         #
#         # # LSTM layer
#         # self.lstm = nn.LSTM(input_size=input_size, num_layers=1, batch_first=True, hidden_size=hidden_layer_size)
#         # self.layernorm = nn.LayerNorm(hidden_layer_size)
#         # self.batchnorm = nn.BatchNorm1d(hidden_layer_size)
#         # self.maxpool = MaxOverTimePooling()  # Hier registrieren
#         #
#         # # Additional fully connected layers
#         # self.fc1 = nn.Linear(hidden_layer_size, fc1_dim // 4)  # Use integer division
#         # self.fc5 = nn.Linear(fc1_dim // 4, self.outputsize)  # Final FC layer for output
#         # self.activation = nn.ELU()
#         #
#         # # Dropout layer
#         # self.dropout = nn.Dropout(p=dropout_rate)
#         # self.to(self.device)
#
#     # def print_model_parameter_size(self):
#     #     total_params = sum(p.numel() for p in self.parameters())
#     #     trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
#     #
#     #     print("CURRENT MODEL IS: ", self.modelName)
#     #     print(f"Total parameters: {total_params}")
#     #     print(f"Trainable parameters: {trainable_params}")
#
#     def forward(self, input_seq, return_intermediates=False):
#         # input_seq = input_seq.to(self.device)  # Ensure input is on the correct device
#         # input_activations = self.input_linear(input_seq)
#         #
#         # # LSTM-Pass (hier verwendest du input_seq oder input_activations – je nach Bedarf)
#         # lstm_out, _ = self.lstm(input_seq)
#         #
#         # # Permutiere, damit die Dimensionen passen: (batch, hidden_layer_size, seq_length)
#         # lstm_out_perm = lstm_out.permute(0, 2, 1)
#         #
#         # # Wende Batch-Normalization an
#         # lstm_out_norm = self.batchnorm(lstm_out_perm)
#         #
#         # # Permutiere zurück zu (batch, seq_length, hidden_layer_size)
#         # lstm_out_3 = lstm_out_norm.permute(0, 2, 1)
#         # lstm_out_max = self.maxpool(lstm_out_3)  # Max über seq_length
#         #
#         # lstm_dropout = self.dropout(lstm_out_max)
#         # fc1_out = self.fc1(lstm_dropout)
#         # fc1_elu_out = self.activation(fc1_out)
#         # predictions = self.fc5(fc1_elu_out)
#         #
#         # if return_intermediates:
#         #     intermediates = {'input_seq': input_seq,
#         #                      'Input_activations': input_activations,
#         #                      'Input_Weights': self.input_linear.weight.data.cpu().numpy(),
#         #                      'LSTM_Out': lstm_out,
#         #                      'LSTM_Weights_IH': self.lstm.weight_ih_l0.data.cpu().numpy(),
#         #                      'LSTM_Weights_HH': self.lstm.weight_hh_l0.data.cpu().numpy(),
#         #                      'Max_Timestep': lstm_out_max,
#         #                      'FC1_Out': fc1_out,
#         #                      'FC1_Weights': self.fc1.weight.data.cpu().numpy(),
#         #                      'FC1_ELU_Out': fc1_elu_out,
#         #                      'Output': predictions,
#         #                      'FC5_Weights': self.fc5.weight.data.cpu().numpy()}
#         #     return predictions, intermediates
#         # else:
#         #     return predictions
#
#         intermediates = {'Input': input_seq}
#
#         lstm_out, _ = self.lstm(input_seq)
#         lstm_out_1 = lstm_out.permute(0, 2, 1)  # Change to (batch_size, num_features, seq_length)
#         lstm_out_2 = self.batchnorm(lstm_out_1)
#         lstm_out_3 = lstm_out_2.permute(0, 2, 1)  # Change back to (batch_size, seq_length, num_features)
#
#         # 🔹 Apply Max Pooling Along the Time Dimension
#         lstm_out_max_timestep = self.maxpool(lstm_out_3.permute(0, 2, 1)).squeeze(2)  # (batch, features)
#
#         lstm_dropout = self.dropout(lstm_out_max_timestep)
#         fc1_out = self.fc1(lstm_dropout)
#         fc1_elu_out = self.activation(fc1_out)
#         predictions = self.fc5(fc1_elu_out)
#
#         if return_intermediates:
#             intermediates['LSTM'] = lstm_out
#             intermediates['Max Timestep'] = lstm_out_max_timestep
#             intermediates['Output FC 1'] = fc1_out
#             intermediates['ELU'] = fc1_elu_out
#             intermediates['Output'] = predictions
#             return predictions, intermediates
#         else:
#             return predictions

import numpy as np
import torch.nn as nn
import torch

class MaxOverTimePooling(nn.Module):
    def __init__(self):
        super(MaxOverTimePooling, self).__init__()

    def forward(self, x):
        return x.max(dim=1)[0]  # Max über die Zeitdimension (seq_length)



class FOVAL(nn.Module):
    def __init__(self, input_size=38, embed_dim=150, dropout_rate=0.5, output_size=1, fc1_dim=32):
        super(FOVAL, self).__init__()
        self.modelName = "Foval"
        self.hidden_layer_size = embed_dim
        self.input_size = input_size
        self.do_inverse_transform = False
        self.target_scaler = None

        # Linear layer to transform input features if needed
        self.input_linear = nn.Linear(in_features=self.input_size, out_features=self.input_size)

        self.dropout = nn.Dropout(p=dropout_rate)

        self.lstm = nn.LSTM(input_size=self.input_size, num_layers=1, batch_first=True,
                            hidden_size=self.hidden_layer_size)
        self.layernorm = nn.LayerNorm(self.hidden_layer_size)
        self.batchnorm = nn.BatchNorm1d(self.hidden_layer_size)

        # 🔹 Explicitly Add MaxPool1d Layer
        self.maxpool = nn.MaxPool1d(kernel_size=10)  # Pool over the entire sequence length

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

        lstm_out, _ = self.lstm(input_seq)
        lstm_out_1 = lstm_out.permute(0, 2, 1)  # Change to (batch_size, num_features, seq_length)
        lstm_out_2 = self.batchnorm(lstm_out_1)
        lstm_out_3 = lstm_out_2.permute(0, 2, 1)  # Change back to (batch_size, seq_length, num_features)

        # 🔹 Apply Max Pooling Along the Time Dimension
        lstm_out_max_timestep = self.maxpool(lstm_out_3.permute(0, 2, 1)).squeeze(2)  # (batch, features)

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
