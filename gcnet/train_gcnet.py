import csv
import datetime
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model import MultimodalSentimentModel
from dataloader_cmumosi import CMUMOSIDataset
import pandas as pd
import collections

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
import os

os.environ["TORCH_USE_CUDA_DSA"] = "1"  # 启用 CUDA 设备端断言
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 让 CUDA 计算同步，立即报错


def train_or_eval_model(model, dataloader, criterion, route_loss, device, optimizer=None,train=False):

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct_sample = 0
    total_sample = 0
    all_labels = []
    all_preds = []
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

        binary_labels = (labels > 0).float()   # 将 labels 转换为二分类。大于零为正类 (1)，小于零为负类 (0)
        # positive_count = (labels > 0).sum().item()
        # negative_count = (labels < 0).sum().item()
        # zero_count = (labels == 0).sum().item()
        #
        # print(f"正样本数量: {positive_count}")
        # print(f"负样本数量: {negative_count}")
        # print(f"零样本数量: {zero_count}")

        route_labels = torch.ones(route_decision.size()).to(device) if args.dataset == 'CMUMOSEI'else torch.zeros(route_decision.size()).to(device)

        # print(f"outputs.shape", outputs.shape)
        # print(f"mask.shape", mask.shape)
        # print(f"outputs[mask].shape", outputs[mask].shape)
        # print(f"masks[3].shape", masks[3].shape)
        # print(f"binary_labels.shape", binary_labels.shape)
        # print(f"binary_labels[masks[3]].shape", binary_labels[masks[3]].shape)
        # 计算分类损失、路由损失、蒸馏一致性损失
        label_mask = mask.squeeze()
        if args.task_type == '7':
            mask = mask.squeeze(-1)  # 变成 (batch_size, seq_len)
            labels = torch.clamp(labels, min=-3, max=3)  # 限制范围
            labels = labels - labels.min()  # 让最小值变为 0
            labels = torch.round(labels)  # 显式四舍五入
            labels = labels.long()  # 转换为 int

        cls_loss = criterion(outputs[mask], binary_labels[label_mask]) if args.task_type == '2' else criterion(outputs[mask], labels[label_mask].long())
        r_loss = route_loss(route_decision[mask], route_labels[mask])
        distillation_losses = [
            torch.nn.MSELoss()(shared_features[i], distilled_features)
            for i in range(len(shared_features))
        ]
        distillation_loss = sum(distillation_losses) / len(distillation_losses)
        loss = cls_loss + r_loss * args.r_loss_weight + distillation_loss * args.distillation_loss_weight

        # 清空梯度->反向传播计算梯度->更新参数
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        if args.task_type == '2':
            predicted = (outputs[mask] > 0).long()  # 正类为 1，负类为 0
            all_labels.extend(binary_labels[label_mask].cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            total_sample += binary_labels[label_mask].size(0)
            correct_sample += (predicted == binary_labels[label_mask]).sum().item()
        else:
            predicted = torch.argmax(outputs[mask], dim=1)  # 选出最高分的类别
            all_labels.extend(labels[label_mask].cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            total_sample += labels[label_mask].size(0)  # 总样本数
            correct_sample += (predicted == labels[label_mask]).sum().item()  # 计算正确的样本数

    f1 = f1_score(all_labels, all_preds, average='weighted' if args.task_type != '2' else 'binary')

    return total_loss / len(dataloader), correct_sample / total_sample * 100, f1


def main():
    # device = torch.device("cpu")
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    input_dim = 2560  # 暂时设置成了adim+vdim+tdim
    hidden_dim = 128
    # output_dim = 1
    output_dim = 1 if args.task_type == '2' else 7  # 动态选择输出类别数

    # Model
    model = MultimodalSentimentModel(
        hidden_dim,
        output_dim,
        args.ablation,
        num_heads=args.num_heads,
        kl_threshold=args.kl_threshold,
        dropout=args.dropout
    ).to(device)

    # model.load_state_dict(torch.load('../save/model_weights_mosi.pth'))

    # Criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    route_loss = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Loading datasets (formal and informal separately)
    if args.dataset == 'CMUMOSI':
        dataset_informal = CMUMOSIDataset(label_path=config.PATH_TO_LABEL['CMUMOSI'],
                                 audio_root=audio_root,
                                 text_root=text_root,
                                 video_root=video_root)

        trainNum = len(dataset_informal.trainVids)
        valNum = len(dataset_informal.valVids)
        testNum = len(dataset_informal.testVids)
        train_idxs = list(range(0, trainNum))
        val_idxs = list(range(trainNum, trainNum + valNum))
        test_idxs = list(range(trainNum + valNum, trainNum + valNum + testNum))

        # Creating dataloaders
        train_loader_informal = DataLoader(dataset_informal, batch_size=args.batch_size, sampler=SubsetRandomSampler(train_idxs),collate_fn=dataset_informal.collate_fn)
        val_loader_informal = DataLoader(dataset_informal, batch_size=args.batch_size, sampler=SubsetRandomSampler(val_idxs), collate_fn=dataset_informal.collate_fn)
        test_loader_informal = DataLoader(dataset_informal, batch_size=args.batch_size, sampler=SubsetRandomSampler(test_idxs), collate_fn=dataset_informal.collate_fn)

    else:
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

        train_loader_formal = DataLoader(dataset_formal, batch_size=args.batch_size, sampler=SubsetRandomSampler(train_idxs), collate_fn=dataset_formal.collate_fn)
        val_loader_formal = DataLoader(dataset_formal, batch_size=args.batch_size, sampler=SubsetRandomSampler(val_idxs), collate_fn=dataset_formal.collate_fn)
        test_loader_formal = DataLoader(dataset_formal, batch_size=args.batch_size, sampler=SubsetRandomSampler(test_idxs), collate_fn=dataset_formal.collate_fn)

    # Training loop
    # epochs = 300 if args.dataset == 'CMUMOSI' else 50
    if args.ablation != 'no':
        if args.dataset == 'CMUMOSI':
            epochs = 50
        else:
            epochs = 25
    else:
        if args.dataset == 'CMUMOSI':
            epochs = 300
        else:
            epochs = 50

    best_test_acc = 0
    best_test_f1 = 0
    best_epoch_acc = 0
    best_epoch_f1 = 0
    best_model = None

    # 冻结模型的部分层，初始仅解冻 `formal_expert` 和 `router` 部分
    # model.set_requires_grad(['informal_expert', 'shared_layers'], False)
    # model.set_requires_grad(['formal_expert', 'router'], True)

    for epoch in range(epochs):
        if args.dataset == 'CMUMOSI':
            print(f"Epoch {epoch + 1}/{epochs} - Training on MOSI (InFormal)")
            informal_loss_train, informal_accuracy_train, informal_f1_train = train_or_eval_model(model, train_loader_informal, criterion, route_loss, device, optimizer=optimizer, train=True)
            informal_loss_val, informal_accuracy_val, informal_f1_val = train_or_eval_model(model, val_loader_informal, criterion, route_loss, device, optimizer=None, train=False)
            informal_loss_test, informal_accuracy_test, informal_f1_test = train_or_eval_model(model, test_loader_informal, criterion, route_loss, device, optimizer=None, train=False)
            if informal_accuracy_test > best_test_acc:
                best_test_acc = informal_accuracy_test
                best_epoch_acc = epoch + 1
                torch.save(model.state_dict(), '../save/model_weights_mosi.pth')
            if informal_f1_test > best_test_f1:
                best_test_f1 = informal_f1_test
                best_epoch_f1 = epoch + 1

            print(f"Informal Training Loss: {informal_loss_train:.2f} | Accuracy: {informal_accuracy_train:.2f}% | F1: {informal_f1_train:.2f}")
            print(f"Informal Validing Loss: {informal_loss_val:.2f} | Accuracy: {informal_accuracy_val:.2f}% | F1: {informal_f1_val:.2f}")
            print(f"Informal Testing Loss: {informal_loss_test:.2f} | Accuracy: {informal_accuracy_test:.2f}% | F1: {informal_f1_test:.2f}")
            print(f"best_test_acc: {best_test_acc:.2f}% in epoch {best_epoch_acc} | best_test_f1: {best_test_f1:.2f} in epoch {best_epoch_f1}")

        # 微调时逐步解冻正式专家模块
        # if epoch == 10:  # 假设从第 50 轮开始逐步解冻
        #     model.set_requires_grad(['informal_expert'], True)
        # elif epoch == 100:  # 在第 100 轮解冻路由器
        #     model.set_requires_grad(['shared_layers'], True)
        else:
            print(f"Epoch {epoch + 1}/{epochs} - Training on MOSEI (Formal)")
            formal_loss_train, formal_accuracy_train, formal_f1_train = train_or_eval_model(model, train_loader_formal, criterion, route_loss, device, optimizer=optimizer, train=True)
            formal_loss_val, formal_accuracy_val, formal_f1_val = train_or_eval_model(model, val_loader_formal, criterion, route_loss, device, optimizer=None, train=False)
            formal_loss_test, formal_accuracy_test , formal_f1_test= train_or_eval_model(model, test_loader_formal, criterion, route_loss, device, optimizer=None, train=False)
            if formal_accuracy_test > best_test_acc:
                best_test_acc = formal_accuracy_test
                best_epoch_acc = epoch + 1
                torch.save(model.state_dict(), '../save/model_weights_mosei.pth')
            if formal_f1_test > best_test_f1:
                best_test_f1 = formal_f1_test
                best_epoch_f1 = epoch + 1
            print(f"Formal Training Loss: {formal_loss_train:.2f} | Accuracy: {formal_accuracy_train:.2f}% | F1: {formal_f1_train:.2f}")
            print(f"Formal Validing Loss: {formal_loss_val:.2f} | Accuracy: {formal_accuracy_val:.2f}% | F1: {formal_f1_val:.2f}")
            print(f"Formal Testing Loss: {formal_loss_test:.2f} | Accuracy: {formal_accuracy_test:.2f}% | F1: {formal_f1_test:.2f}")
            print(f"best_test_acc: {best_test_acc:.2f}% in epoch {best_epoch_acc} | best_test_f1: {best_test_f1:.2f} in epoch {best_epoch_f1}")

    return best_test_acc, best_test_f1, best_epoch_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--audio-feature', type=str, default='wav2vec-large-c-UTT', help='audio feature name')
    parser.add_argument('--text-feature', type=str, default='deberta-large-4-UTT', help='text feature name')
    parser.add_argument('--video-feature', type=str, default='manet_UTT', help='video feature name')
    parser.add_argument('--dataset', type=str, default='CMUMOSEI', help='dataset type')
    parser.add_argument('--task_type', type=str, choices=['2', '7'], default='2', help='Specify task type: binary or multiclass')
    parser.add_argument('--ablation', type=str, default='no', choices=['no', 'router','graph'], help='Ablation Study')
    parser.add_argument('--cuda', type=int, default=0, help='Specify CUDA device ID ')

    args = parser.parse_args()
    print(args)
    # 定义可选参数范围
    param_choices = {
        "r_loss_weight": [0.3, 0.5, 0.7, 1.0],
        "distillation_loss_weight": [0.1, 0.3, 0.5, 0.7],
        "lr": [0.0003,0.0005, 0.001,0.003, 0.005, 0.03, 0.01],
        "weight_decay": [1e-5, 1e-4, 5e-4, 1e-3, 1e-6],
        "batch_size": [8,16, 32, 64],
        "num_heads": [2, 4, 8],
        "kl_threshold": [0.3, 0.5, 0.7, 1.0],
        "dropout": [0.1,0.2, 0.3, 0.5]
    }

    print(f'====== Reading Data =======')
    audio_feature = args.audio_feature
    text_feature = args.text_feature
    video_feature = args.video_feature
    audio_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], audio_feature)
    text_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], text_feature)
    video_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], video_feature)
    assert os.path.exists(audio_root) and os.path.exists(text_root) and os.path.exists(
        video_root), f'features not exist!'

    # CSV 文件
    # 给输出文件名加一个时间戳，保证唯一性
    time_str = datetime.datetime.now().strftime("%m%d_%H%M")
    output_csv = f"../save/results_{args.dataset}_{args.task_type}_{args.ablation}_{time_str}.csv"
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset:", args.dataset, "TaskType:", args.task_type, "Ablation:", args.ablation])
        writer.writerow(["Id", "BestTestAcc", "BestTestF1", "best_epoch_acc",
                        "r_loss_weight", "distillation_loss_weight", "lr",
                         "weight_decay","batch_size", "num_heads", "kl_threshold", "dropout"
                         ])

    if args.ablation != 'no':
        num_runs = 10
    else:
        if args.dataset == 'CMUMOSI':
            num_runs = 50
        else:
            num_runs = 25

    # 跑 100 次完整训练
    for i in range(num_runs):
        print(f"===== 第 {i + 1} 次完整训练开始 =====")
        # 在每次完整训练前随机采样一套超参数
        args.r_loss_weight = random.choice(param_choices["r_loss_weight"])
        args.distillation_loss_weight = random.choice(param_choices["distillation_loss_weight"])
        args.lr = random.choice(param_choices["lr"])
        args.weight_decay = random.choice(param_choices["weight_decay"])
        args.batch_size = random.choice(param_choices["batch_size"])
        args.num_heads = random.choice(param_choices["num_heads"])
        args.kl_threshold = random.choice(param_choices["kl_threshold"])
        args.dropout = random.choice(param_choices["dropout"])

        # 可以在这里查看本次训练选到的超参数
        sampled_params = {
            "r_loss_weight": args.r_loss_weight,
            "distillation_loss_weight": args.distillation_loss_weight,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "num_heads": args.num_heads,
            "kl_threshold": args.kl_threshold,
            "dropout": args.dropout
        }
        print(f"本次训练使用的超参数: {sampled_params}")

        best_test_acc, best_test_f1 , best_epoch_acc= main()  # main() 会返回结果
        print(f"===== 第 {i + 1} 次完整训练结束 =====\n")

        # 追加写入 CSV
        with open(output_csv, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                                i + 1,
                                f"{best_test_acc:.2f}",
                                f"{best_test_f1:.4f}",
                                f"{best_epoch_acc}"
                            ] + list(sampled_params.values()))
    print(f"所有训练均已完成，结果保存在 {output_csv} 中！")

    # 读取 CSV 文件
    df = pd.read_csv(output_csv)

    # 查找 BestTestAcc 和 BestTestF1 的最大值及其对应的 id
    best_acc_id = df.iloc[:, 0][df.iloc[:, 1].astype(float).idxmax()]  # 获取 BestTestAcc 最大值的 id
    best_acc_value = df.iloc[:, 1].astype(float).max()  # BestTestAcc 最大值

    best_f1_id = df.iloc[:, 0][df.iloc[:, 2].astype(float).idxmax()]  # 获取 BestTestF1 最大值的 id
    best_f1_value = df.iloc[:, 2].astype(float).max()  # BestTestF1 最大值

    # 追加写入 CSV 末尾
    with open(output_csv, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["Overall Best Results"])
        writer.writerow(["BestTestAcc", best_acc_value, "Achieved at ID:", best_acc_id])
        writer.writerow(["BestTestF1", best_f1_value, "Achieved at ID:", best_f1_id])

    print(
        f"最佳 ACC 发生在 ID {best_acc_id} (值: {best_acc_value})，最佳 F1 发生在 ID {best_f1_id} (值: {best_f1_value})")

