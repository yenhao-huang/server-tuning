# From dalinvip/cnn-lstm-bilstm-clstm (models/model_LSTM.py)
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Neural Networks model : LSTM
"""


class RNN_Classifier(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_classes,
        n_counters,
        num_layers=2,
        dropout=0.4,
    ):
        super(RNN_Classifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        C = num_classes

        # lstm
        self.lstm = nn.LSTM(
            n_counters,
            hidden_dim,
            dropout=dropout,
            num_layers=num_layers,
            batch_first=True,
        )

        # linear
        self.hidden2label = nn.Linear(hidden_dim, C)

    def forward(self, x):
        # lstm
        lstm_out, _ = self.lstm(x)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        # pooling
        lstm_out = F.tanh(lstm_out)
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
        lstm_out = F.tanh(lstm_out)
        # linear
        logit = self.hidden2label(lstm_out)
        return logit
