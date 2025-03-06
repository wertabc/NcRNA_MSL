import pandas as pd
import torch
import torch.nn as nn
import math
import torch.nn.functional as F



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1) :
        super(PositionalEncoding, self).__init__()
#         实现Dropout正则化减少过拟合
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self,word_list, text_len, embedding_dim):
        """
        :param text_len: 序列长度
        :param word_list:
        :param embedding_dim:
        """

        super().__init__()
        self.em = nn.Embedding(len(word_list) + 1, embedding_dim=50)  # 对0也需要编码
        self.pos = PositionalEncoding(embedding_dim, max_len=text_len)
        self.encoder_layer = nn.TransformerEncoderLayer(embedding_dim, nhead=5)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.pool1 = nn.MaxPool1d(5)
        self.pool2 = nn.MaxPool1d(3)
        self.pool3 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(text_len, 500)
        self.fc2 = nn.Linear(500, 128)
        self.fc3 = nn.Linear(128, 2)


    def forward(self, inputs):
        x = self.em(inputs)                                  #[16,1000,50]
        x = self.pos(x)                                     #[16,1000,50]
        x = x.float()                                       #[16,1000,50]
        x = self.transformer_encoder(x)                     #[16,1000,50]
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)
        x = x.squeeze(-1)                          # [1000,50]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))


        return self.fc3(x)