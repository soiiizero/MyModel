import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model import MultimodalSentimentModel  # Importing the newly created model
from dataloader_cmumosi import CMUMOSIDataset  # Assuming the dataset class remains the same for MOSI and MOSEI

import os
import time
import glob
import math
import pickle
import random
import argparse
import numpy as np
from numpy.random import randint

from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder

import sys

sys.path.append('../')
import config

def train_or_eval_model(model, dataloader, criterion,  device, optimizer=None,train=False):

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct_sample = 0
    total_sample = 0
    for data in dataloader:

        # print(data[3])
        # print(type(data)) # list
        # print(type(data[3])) # torch.Tensor
        # for i, tensor in enumerate(data[3]):
        #     print(f"Tensor {i} shape: {tensor.shape}") # Tensor 31 shape: torch.Size([44])
        # print(data[0].shape)  # torch.Size([32, seq, 512]) [seqlen, batch, dim]??seqlen会变因为每个批次中样本的最大片段数不一样
        # print(data[1].shape)  # torch.Size([32, seq, 1024])
        # print(data[2].shape)  # torch.Size([32, seq, 1024])
        # print(data[3].shape)  # torch.Size([32, seq])

        inputs = torch.cat((data[0], data[1], data[2]), dim=-1)
        # print(f"inputs形状：{inputs.shape}")   # torch.Size([32, seq, 2560])
        labels = data[3]
        labels = labels.view(-1)
        # print(f"labels形状：{labels.shape}")
        inputs, labels = inputs.to(device), labels.to(device)
        context = torch.zeros(1, 256).to(device)

        # 前向传播计算预测值
        outputs = model(inputs, context)
        outputs = outputs.view(-1, outputs.size(-1))
        # print(f"outputs形状：{outputs.shape}")
        binary_labels = (labels > 0).long()  # 将 labels 转换为二分类。大于零为正类 (1)，小于零为负类 (0)

        # 计算损失
        loss = criterion(outputs, binary_labels)

        # 清空梯度->反向传播计算梯度->更新参数
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total_sample += labels.size(0)
        correct_sample += (predicted == labels).sum().item()

    return total_loss / len(dataloader), correct_sample / total_sample * 100


def main():
    # device = torch.device("cpu")

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    input_dim = 2560  # 暂时设置成了adim+vdim+tdim
    hidden_dim = 128
    output_dim = 2  # Assuming binary sentiment classification

    # Instantiate the model
    model = MultimodalSentimentModel(input_dim, hidden_dim, output_dim).to(device)

    # Criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Loading datasets (formal and informal separately)

    # dataset_informal = CMUMOSIDataset(label_path=config.PATH_TO_LABEL['CMUMOSI'],
    #                          audio_root=audio_root,
    #                          text_root=text_root,
    #                          video_root=video_root)
    dataset_formal = CMUMOSIDataset(label_path=config.PATH_TO_LABEL['CMUMOSEI'],
                             audio_root=audio_root,
                             text_root=text_root,
                             video_root=video_root)

    trainNum = len(dataset_formal.trainVids)
    valNum = len(dataset_formal.valVids)
    testNum = len(dataset_formal.testVids)
    train_idxs = list(range(0, trainNum))
    val_idxs = list(range(trainNum, trainNum + valNum))
    test_idxs = list(range(trainNum + valNum, trainNum + valNum + testNum))

    # Creating dataloaders
    # train_loader_informal = DataLoader(dataset_informal, batch_size=32, shuffle=True, collate_fn=dataset_informal.collate_fn)
    train_loader_formal = DataLoader(dataset_formal, batch_size=32, sampler=SubsetRandomSampler(train_idxs), collate_fn=dataset_formal.collate_fn)
    val_loader_formal = DataLoader(dataset_formal, batch_size=32, sampler=SubsetRandomSampler(val_idxs), collate_fn=dataset_formal.collate_fn)
    test_loader_formal = DataLoader(dataset_formal, batch_size=32, sampler=SubsetRandomSampler(test_idxs), collate_fn=dataset_formal.collate_fn)

    # Training loop
    epochs = 20
    for epoch in range(epochs):

        # print(f"Epoch {epoch + 1}/{epochs} - Training on MOSI (Informal)")
        # informal_loss, informal_accuracy = train(model, train_loader_informal, criterion, optimizer, device)
        # print(f"Informal Training Loss: {informal_loss:.4f} | Accuracy: {informal_accuracy:.2f}%")

        print(f"Epoch {epoch + 1}/{epochs} - Training on MOSEI (Formal)")
        formal_loss_train, formal_accuracy_train = train_or_eval_model(model, train_loader_formal, criterion, device, optimizer=optimizer, train=True)
        formal_loss_val, formal_accuracy_val = train_or_eval_model(model, val_loader_formal, criterion, device, optimizer=None, train=False)
        formal_loss_test, formal_accuracy_test = train_or_eval_model(model, test_loader_formal, criterion, device, optimizer=None, train=False)

        print(f"Formal Training Loss: {formal_loss_train:.4f} | Accuracy: {formal_accuracy_train:.2f}%")
        print(f"Formal Validing Loss: {formal_loss_val:.4f} | Accuracy: {formal_accuracy_val:.2f}%")
        print(f"Formal Testing Loss: {formal_loss_test:.4f} | Accuracy: {formal_accuracy_test:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Params for input
    parser.add_argument('--audio-feature', type=str, default='wav2vec-large-c-UTT', help='audio feature name')
    parser.add_argument('--text-feature', type=str, default='deberta-large-4-UTT', help='text feature name')
    parser.add_argument('--video-feature', type=str, default='manet_UTT', help='video feature name')
    parser.add_argument('--dataset', type=str, default='CMUMOSEI', help='dataset type')

    args = parser.parse_args()
    print(args)

    print(f'====== Reading Data =======')
    audio_feature = args.audio_feature
    text_feature = args.text_feature
    video_feature = args.video_feature
    audio_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], audio_feature)
    text_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], text_feature)
    video_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], video_feature)
    assert os.path.exists(audio_root) and os.path.exists(text_root) and os.path.exists(
        video_root), f'features not exist!'

    main()
