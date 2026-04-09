import torch
from torch import nn

class ParallelCNN(nn.Module):
    def __init__(self, para_ker, pool_kernel=6, drop=0.5):
        super().__init__()
        self.lseq = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(4, 4, kernel_size=k, padding="same"),
                nn.ReLU(),
                nn.MaxPool1d(pool_kernel),
                nn.Dropout(drop)
            ) for k in para_ker
        ])

    def forward(self, inputs):
        return torch.cat([seq(inputs) for seq in self.lseq], dim=1)


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, inputs):
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(inputs)
        return self.linear(recurrent)


class GPlusD(nn.Module):
    def __init__(self, para_ker, input_shape=(64, 300, 4), pool_kernel=6, drop=0.5):
        super().__init__()
        binode = len(para_ker) * 4

        self.pconv = ParallelCNN(para_ker, pool_kernel, drop)
        self.bilstm = BidirectionalLSTM(binode, binode, binode)
        self.flatten = nn.Flatten()
        
        flat_dim = self._get_feature_shape(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(flat_dim, flat_dim),
            nn.ReLU(),
            nn.Linear(flat_dim, 2),
            nn.Softmax(dim=1)
        )

    def _get_feature_shape(self, shape):
        with torch.no_grad():
            x = torch.zeros(shape).permute(0, 2, 1)
            x = self.pconv(x).permute(0, 2, 1)
            x = self.bilstm(x)
            return self.flatten(x).shape[1]

    def forward(self, x):
        x = self.pconv(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.bilstm(x)
        return self.fc(self.flatten(x))