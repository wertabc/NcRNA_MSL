import math
import sys
import re
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torch.nn import functional as F

class Net(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super(Net, self).__init__()
        self.em = nn.Embedding(vocab_size + 1, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_size)
        self.pool1 = nn.MaxPool1d(5)
        self.pool2 = nn.MaxPool1d(3)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, inputs):
        bz = inputs.shape[1]
        device = inputs.device
        h0 = torch.zeros((1, bz, self.rnn.hidden_size)).to(device)  # Corrected hidden_size
        c0 = torch.zeros((1, bz, self.rnn.hidden_size)).to(device)  # Corrected hidden_size
        x = self.em(inputs)                                         # [seq_len, batch, embed_dim]
        r_o, _ = self.rnn(x, (h0, c0))                              # [seq_len, batch, hidden_size]
        # x = self.pool1(r_o.permute(1, 2, 0))                        # [batch, hidden_size, seq_len] -> [batch, hidden_size, new_seq_len]
        x = self.pool1(x)
        x = self.pool2(x)                                            # [batch, hidden_size, new_seq_len2] -> [batch, hidden_size, new_seq_len3]
        x = x.view(x.size(0), -1)                                   # Flatten
        # x = x.squeeze(-1)
        x = F.dropout(F.relu(self.fc1(x)))                          # [batch, hidden_size] -> [batch, 128]
        x = self.fc2(x)                                             # [batch, 128] -> [batch, 2]
        return x















# with torch.no_grad():
#     outputs=[]
#     for x,y in data_dl:
#         out = model(x)
#         # print(out.shape)
#         # sys.exit()
#         outputs.append(out)
#     print(outputs)
#     print(outputs[0].shape)
#     print(len(outputs))
# #r1 = [i.reshape(16000,1) for i in out]
# outputs1 = torch.concat(outputs,axis =0)
# # print(outputs1)
#
# deep_fea_LSTM=pd.DataFrame(outputs1)
# # print(deep_fea_LSTM)
# deep_fea_LSTM.to_csv(r"D:\human_and_mouse\human\lstm_1d18.csv")