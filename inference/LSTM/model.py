import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, output_dim, layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.layers = layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers,
                            #dropout = 0.3,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim*seq_len, output_dim, bias = True) 
        
    # reset h
    def reset_hidden_state(self): 
        # hidden state,cell state
        self.hidden = (
                torch.zeros(self.layers, self.seq_len, self.hidden_dim),
                torch.zeros(self.layers, self.seq_len, self.hidden_dim))
    
    def forward(self, x):
        x, _status = self.lstm(x)
        x=torch.stack([torch.cat([hx for hx in h])for h in x])
        x=self.fc(x)
        return x

