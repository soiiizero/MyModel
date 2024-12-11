import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model import MultimodalSentimentModel
from dataloader_cmumosi import CMUMOSIDataset

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

def train_or_eval_model(model, dataloader, criterion, route_loss, device, optimizer=None,train=False):

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct_sample = 0
    total_sample = 0
    for data, masks in dataloader:

        # print(data[3])
        # print(type(data)) # list
        # print(type(data[3])) # torch.Tensor
        # for i, tensor in enumerate(data[3]):
        #     print(f"Tensor {i} shape: {tensor.shape}") # Tensor 31 shape: torch.Size([44])
        # print(data[0].shape)  # torch.Size([32, seq, 512]) [seqlen, batch, dim]??seqlen会变因为每个批次中样本的最大片段数不一样
        # print(data[1].shape)  # torch.Size([32, seq, 1024])
        # print(data[2].shape)  # torch.Size([32, seq, 1024])
        # print(data[3].shape)  # torch.Size([32, seq])

        inputs = [data[0].to(device), data[1].to(device), data[2].to(device)]
        # print(f"inputs形状：{inputs.shape}")   # torch.Size([32, seq, 2560])
        labels = data[3]
        labels = labels.view(-1)
        masks[3] = masks[3].view(-1)
        # print(f"labels形状：{labels.shape}")
        labels = labels.to(device)
        context = torch.cat((data[0], data[1], data[2]), dim=-1).to(device)
        mask = torch.cat((masks[0], masks[1], masks[2]), dim=-1).to(device)
        # print("mask shape:", mask.shape)
        # 前向传播计算预测值
        outputs, route_decision, shared_features, private_features, distilled_features = model(inputs, context)

        route_decision = route_decision.view(-1, route_decision.size(-1))
        # print(route_decision.shape) # torch.Size([32, seq, 1])
        # print(route_decision[0, :, 0])
        # print(route_decision.mean(dim=1).view(-1))
        outputs = outputs.view(-1, outputs.size(-1))
        route_mask = mask
        mask = mask.view(-1, mask.size(-1))
        mask = mask.any(dim=1, keepdim=True)  # 生成形状为 (num, 1) 的样本级 mask

        # print(f"outputs形状：{outputs.shape}")

        binary_labels = (labels > 0).float()  # 将 labels 转换为二分类。大于零为正类 (1)，小于零为负类 (0)
        # positive_count = (labels > 0).sum().item()
        # negative_count = (labels < 0).sum().item()
        # zero_count = (labels == 0).sum().item()
        #
        # print(f"正样本数量: {positive_count}")
        # print(f"负样本数量: {negative_count}")
        # print(f"零样本数量: {zero_count}")

        route_labels = torch.ones(route_decision.size()).to(device)

        # print(f"outputs.shape", outputs.shape)
        # print(f"mask.shape", mask.shape)
        # print(f"outputs[mask].shape", outputs[mask].shape)
        # print(f"masks[3].shape", masks[3].shape)
        # print(f"binary_labels.shape", binary_labels.shape)
        # print(f"binary_labels[masks[3]].shape", binary_labels[masks[3]].shape)
        # 计算分类损失、路由损失、蒸馏一致性损失
        label_mask = mask.squeeze()
        cls_loss = criterion(outputs[mask], binary_labels[label_mask])
        r_loss = route_loss(route_decision[mask], route_labels[mask])
        distillation_losses = [
            torch.nn.MSELoss()(shared_features[i], distilled_features)
            for i in range(len(shared_features))
        ]
        distillation_loss = sum(distillation_losses) / len(distillation_losses)
        loss = cls_loss + r_loss * 0.5 + distillation_loss * 0.3

        # 清空梯度->反向传播计算梯度->更新参数
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        predicted = (outputs[mask] > 0).long()  # 正类为 1，负类为 0
        total_sample += binary_labels[label_mask].size(0)
        correct_sample += (predicted == binary_labels[label_mask]).sum().item()

    return total_loss / len(dataloader), correct_sample / total_sample * 100


def main():
    # device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    input_dim = 2560  # 暂时设置成了adim+vdim+tdim
    hidden_dim = 128
    output_dim = 1

    # Model
    model = MultimodalSentimentModel(hidden_dim, output_dim).to(device)
    # model.load_state_dict(torch.load('../save/model_weights_mosi.pth'))

    # Criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    route_loss = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Loading datasets (formal and informal separately)
    # dataset_informal = CMUMOSIDataset(label_path=config.PATH_TO_LABEL['CMUMOSI'],
    #                          audio_root=audio_root,
    #                          text_root=text_root,
    #                          video_root=video_root)
    #
    # trainNum = len(dataset_informal.trainVids)
    # valNum = len(dataset_informal.valVids)
    # testNum = len(dataset_informal.testVids)
    # train_idxs = list(range(0, trainNum))
    # val_idxs = list(range(trainNum, trainNum + valNum))
    # test_idxs = list(range(trainNum + valNum, trainNum + valNum + testNum))
    #
    # # Creating dataloaders
    # train_loader_informal = DataLoader(dataset_informal, batch_size=32, sampler=SubsetRandomSampler(train_idxs),collate_fn=dataset_informal.collate_fn)
    # val_loader_informal = DataLoader(dataset_informal, batch_size=32, sampler=SubsetRandomSampler(val_idxs), collate_fn=dataset_informal.collate_fn)
    # test_loader_informal = DataLoader(dataset_informal, batch_size=32, sampler=SubsetRandomSampler(test_idxs), collate_fn=dataset_informal.collate_fn)


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

    train_loader_formal = DataLoader(dataset_formal, batch_size=32, sampler=SubsetRandomSampler(train_idxs), collate_fn=dataset_formal.collate_fn)
    val_loader_formal = DataLoader(dataset_formal, batch_size=32, sampler=SubsetRandomSampler(val_idxs), collate_fn=dataset_formal.collate_fn)
    test_loader_formal = DataLoader(dataset_formal, batch_size=32, sampler=SubsetRandomSampler(test_idxs), collate_fn=dataset_formal.collate_fn)

    # Training loop
    epochs = 1000
    best_test_acc = 0
    best_epoch = 0
    best_model = None

    # 冻结模型的部分层，初始仅解冻 `formal_expert` 和 `router` 部分
    # model.set_requires_grad(['informal_expert', 'shared_layers'], False)
    # model.set_requires_grad(['formal_expert', 'router'], True)

    for epoch in range(epochs):

        # print(f"Epoch {epoch + 1}/{epochs} - Training on MOSI (InFormal)")
        # informal_loss_train, informal_accuracy_train = train_or_eval_model(model, train_loader_informal, criterion, route_loss, device, optimizer=optimizer, train=True)
        # informal_loss_val, informal_accuracy_val = train_or_eval_model(model, val_loader_informal, criterion, route_loss, device, optimizer=None, train=False)
        # informal_loss_test, informal_accuracy_test = train_or_eval_model(model, test_loader_informal, criterion, route_loss, device, optimizer=None, train=False)
        # if informal_accuracy_test > best_test_acc:
        #     best_test_acc = informal_accuracy_test
        #     best_epoch = epoch + 1
        #     torch.save(model.state_dict(), '../save/model_weights_mosi.pth')
        #
        # print(f"Formal Training Loss: {informal_loss_train:.2f} | Accuracy: {informal_accuracy_train:.2f}%")
        # print(f"Formal Validing Loss: {informal_loss_val:.2f} | Accuracy: {informal_accuracy_val:.2f}%")
        # print(f"Formal Testing Loss: {informal_loss_test:.2f} | Accuracy: {informal_accuracy_test:.2f}%")
        # print(f"best_test_acc: {best_test_acc:.2f}% in epoch {best_epoch}")

        # 微调时逐步解冻正式专家模块
        # if epoch == 10:  # 假设从第 50 轮开始逐步解冻
        #     model.set_requires_grad(['informal_expert'], True)
        # elif epoch == 100:  # 在第 100 轮解冻路由器
        #     model.set_requires_grad(['shared_layers'], True)

        print(f"Epoch {epoch + 1}/{epochs} - Training on MOSEI (Formal)")
        formal_loss_train, formal_accuracy_train = train_or_eval_model(model, train_loader_formal, criterion, route_loss, device, optimizer=optimizer, train=True)
        formal_loss_val, formal_accuracy_val = train_or_eval_model(model, val_loader_formal, criterion, route_loss, device, optimizer=None, train=False)
        formal_loss_test, formal_accuracy_test = train_or_eval_model(model, test_loader_formal, criterion, route_loss, device, optimizer=None, train=False)
        if formal_accuracy_test > best_test_acc:
            best_test_acc = formal_accuracy_test
            best_epoch = epoch + 1
            # torch.save(model.state_dict(), '../save/model_weights.pth')
        print(f"Formal Training Loss: {formal_loss_train:.2f} | Accuracy: {formal_accuracy_train:.2f}%")
        print(f"Formal Validing Loss: {formal_loss_val:.2f} | Accuracy: {formal_accuracy_val:.2f}%")
        print(f"Formal Testing Loss: {formal_loss_test:.2f} | Accuracy: {formal_accuracy_test:.2f}%")
        print(f"best_test_acc: {best_test_acc:.2f}% in epoch {best_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
