import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score,matthews_corrcoef
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import sys

def metrics(y_true, y_pred):
    TN = np.count_nonzero(np.logical_and(y_true == 0, y_pred == 0))  # 计算真阴性
    FN = np.count_nonzero(np.logical_and(y_true == 1, y_pred == 0))   # 计算假阴性
    TP = np.count_nonzero(np.logical_and(y_true == 1, y_pred == 1))  # 计算真阳性
    FP = np.count_nonzero(np.logical_and(y_true == 0, y_pred == 1))  # 计算假阳性
    ppv = TP / (TP + FP) if (TP + FP) != 0 else 0.0  # 计算阳性预测值
    npv = TN / (TN + FN) if (TN + FN) != 0 else 0.0  # 计算阴性预测值
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0.0  # 计算特异性
    return ppv,npv,specificity

# 验证函数
def evaluate_model(model, data_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            # predicted = torch.round(outputs)
            all_labels.extend(labels.cpu().detach().numpy())
            all_preds.extend(predicted.cpu().detach().numpy())
            # print(all_labels)
            # print(all_preds)
            # sys.exit()
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    ppv, npv, specificity = metrics(all_labels, all_preds)
    auroc = roc_auc_score(all_labels, all_preds)

    return acc, f1, mcc, recall, specificity, npv, ppv, auroc

# 画图函数

def picture(x_data, y_data, x_label, y_label, title, name, path="/pictures"):
    """
    :param x_data: x轴数据
    :param y_data: y轴数据
    :param x_label: x轴标题
    :param y_label: y轴标题
    :param title: 图像标题
    :param path: 保存路径，默认为当前目录下的pictures文件夹
    :param name: 图像文件名，默认为plot.png
    """
    if not os.path.exists(path):
        # 创建保存路径
        os.makedirs(path)

    # 画折线图
    plt.plot(x_data, y_data)

    # 添加标题和标签
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    name = f"plot_{name}.png"
    # 保存图形到指定目录
    # save_path = os.path.join(path, name)
    # plt.savefig(save_path)

    # 显示图形
    plt.show()


# 训练模型
def train_(model, train_loader, val_loader, criterion, optimizer, name, device, epochs):
    '''
    :param model:
    :param train_loader:
    :param val_loader:
    :param criterion:
    :param optimizer:
    :param device:
    :param epochs:
    :param name: 数据名称
    :return:
    '''

    average_loss = 0
    train_loss = []
    num_batch = len(train_loader)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        model.to(device)
        batch_progress = tqdm(total=num_batch, desc='', unit='batch',position=0,bar_format='{l_bar}{r_bar}')
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_progress.update(1)
        batch_progress.close()
        average_loss = total_loss / num_batch
        train_loss.append(average_loss)
        tqdm.write(f"Epoch: {epoch+1}/{epochs}, Loss: {average_loss:.4f},")


    pic = picture(range(1, epoch+2), train_loss, "Epoch", "Loss", "Train Loss_Plant_Transformer", name)
    plt.savefig(f".src1/pictures_Trainloss/plot_{name}.png")
    pic.close()
    if not os.path.exists('../par'):
        os.makedirs('../par')
    # torch.save(model, f"/par/model_{name}.pth")

    average_val_loss = 0
    val_loss = []
    num_batch = len(val_loader)
    for epoch in range(epochs):
        model.train()
        total_val_loss = 0
        model.to(device)
        batch_progress = tqdm(total=num_batch, desc='', unit='batch', position=0, bar_format='{l_bar}{r_bar}')
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_progress.update(1)
        batch_progress.close()
        average_val_loss = total_val_loss / num_batch
        train_loss.append(average_val_loss)
        tqdm.write(f"Epoch: {epoch + 1}/{epochs}, Loss: {average_val_loss:.4f},")
        pic = picture(range(1, epoch + 2), train_loss, "Epoch", "Loss", "Val Loss Plant_Transformer", name)
        pic.savefig(f".src1/pictures_Valloss/plot_{name}.png")
        pic.close()


    return model

