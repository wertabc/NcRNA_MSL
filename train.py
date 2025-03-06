from src1.dataDeal import data
from src1.transform import Transformer
from src1.function import train_, evaluate_model
import torch

def main1(list_of_args: list):
    """
    :param list_of_args: 配置项
    :return:
    """

    for i in list_of_args:
        paths = i['path2']
        # 数据处理
        data_args = (i["path1"], paths, i["text_len"], i["batch_size"])
        train_dl, val_dl, test_dl, word_list1,word_list2 = data(*data_args)
        # 初始化transform模型
        model = Transformer(word_list1, i["text_len"], i["embedding_dim"])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model = train_(model, train_dl, val_dl, i["criterion"], optimizer, i["name"], i["device"], i["epochs"])

        acc, f1, mcc, recall, specificity, npv, ppv, auroc = evaluate_model(model, test_dl, i["device"])
        print(f"name:{i['name']}"
              f"acc:{acc}, f1:{f1}, mcc:{mcc}, recall:{recall}, specificity:{specificity}, npv:{npv}, ppv:{ppv}, auroc:{auroc}")
