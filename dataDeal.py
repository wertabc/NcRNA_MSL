import re
import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

pat = re.compile('[agctagct]')

class mydataset(torch.utils.data.Dataset):
    def __init__(self, text_list, label_list):
        self.text_list = text_list
        self.label_list = label_list

    def __getitem__(self, index):
        text = torch.LongTensor(self.text_list[index])
        label = self.label_list[index]
        return text, label

    def __len__(self):
        return len(self.text_list)

def pre_process(text):
    return [each.lower() for each in pat.findall(text)]


def data(path1,path2,text_len, batch_size):
    data1 = pd.read_excel(path1)



    #训练集和验证集
    sequences1 = data1['Sequence']
    labels1 = data1['label'].values

    x1 = sequences1.apply(pre_process)

    word_set1 = set()

    for lst in x1:
        for word in lst:
            word_set1.add(word)

    word_list1 = list(word_set1)
    word_index1 = dict([(each, word_list1.index(each) + 1) for each in word_list1])
    text1 = x1.apply(lambda x1: [word_index1.get(word, 0) for word in x1])

    pad_text1 = [l + (text_len - len(l)) * [0] if len(l) < text_len else l[:text_len] for l in text1]
    pad_text1 = np.array(pad_text1)

    pad_text1, labels1 = torch.LongTensor(pad_text1), torch.LongTensor(labels1)
    x_train, x_val, y_train, y_val = train_test_split(pad_text1, labels1, test_size=0.2)

    #测试集
    test_loaders = []
    for test_path in path2:
        test_data = pd.read_csv(test_path)
        test_sequences = test_data['Sequence']
        test_labels = test_data['label'].values
        x2 = test_sequences.apply(pre_process)

        word_set2 = set()

        for lst in x2:
            for word in lst:
                word_set2.add(word)

        word_list2 = list(word_set2)
        word_index2 = dict([(each, word_list2.index(each) + 1) for each in word_list2])
        text = x2.apply(lambda x2: [word_index2.get(word, 0) for word in x2])

        test_pad_text = [l + (text_len - len(l)) * [0] if len(l) < text_len else l[:text_len] for l in text]
        test_pad_text = np.array(test_pad_text)
        test_pad_text, test_labels = torch.LongTensor(test_pad_text), torch.LongTensor(test_labels)

    train_ds = mydataset(x_train, y_train)
    val_ds = mydataset(x_val, y_val)
    test_ds = mydataset(test_pad_text, test_labels)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    return train_dl, val_dl, test_dl, word_list1,word_list2
