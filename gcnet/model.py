import torch
import torch.nn as nn

class FeatureDecoupler(nn.Module):
    def __init__(self, feature_dims, shared_dims, private_dims):
        super(FeatureDecoupler, self).__init__()
        self.num_modalities = len(feature_dims)
        self.shared_fc = nn.ModuleList([
            nn.Linear(feature_dims[i], shared_dims[i]) for i in range(self.num_modalities)
        ])
        self.private_fc = nn.ModuleList([
            nn.Linear(feature_dims[i], private_dims[i]) for i in range(self.num_modalities)
        ])

    def forward(self, features):
        shared_features = [self.shared_fc[i](features[i]) for i in range(self.num_modalities)]
        private_features = [self.private_fc[i](features[i]) for i in range(self.num_modalities)]
        return shared_features, private_features

class DynamicGraphDistiller(nn.Module):
    def __init__(self, feature_dims, output_dim=128):
        super(DynamicGraphDistiller, self).__init__()
        self.num_modalities = len(feature_dims)
        self.feature_dims = feature_dims
        self.dynamic_weights = nn.Parameter(torch.rand(self.num_modalities, self.num_modalities))
        self.proj_layers = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim) for feature_dim in feature_dims
        ])
        self.output_proj = nn.Linear(feature_dims[0], output_dim)

    def forward(self, features):
        # 投影每个模态特征
        projected_features = [
            proj(features[i]) for i, proj in enumerate(self.proj_layers)
        ]
        # 动态加权组合
        batch_size, seq_len, _ = projected_features[0].shape
        distilled_features = torch.zeros(batch_size, seq_len, self.feature_dims[0], device=features[0].device)
        for i in range(self.num_modalities):
            for j in range(self.num_modalities):
                distilled_features += self.dynamic_weights[i][j] * projected_features[j]
        # 投影到目标维度
        distilled_features = self.output_proj(distilled_features)
        return distilled_features


class RoutingMechanism(nn.Module):
    def __init__(self, input_dim=2560, hidden_dim=128, output_dim=1, num_heads=4, dropout_rate=0.3):
        super(RoutingMechanism, self).__init__()
        # 增加多头注意力层
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout_rate)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, context):
        context, _ = self.attention(context, context, context)
        context = self.fc1(context)
        context = self.relu(context)
        context = self.dropout(context)
        context = self.fc2(context)
        return torch.sigmoid(context)  # 输出在0到1之间，用于动态加权


class InformalExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, dropout_rate=0.3):
        super(InformalExpert, self).__init__()
        # 使用多头自注意力增强对模态特征的捕捉
        # self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout_rate)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        # x, _ = self.attention(x, x, x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class FormalExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, dropout_rate=0.3):
        super(FormalExpert, self).__init__()
        # 使用多头自注意力增强对模态特征的捕捉
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout_rate)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x, _ = self.attention(x, x, x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MultimodalSentimentModel(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        """
        多模态情感分析模型。
        参数:
            feature_dims: 每个模态的输入维度列表，如 [512, 1024, 1024]。
            shared_dims: 每个模态的无关特征维度列表，如 [128, 128, 128]。
            private_dims: 每个模态的特有特征维度列表，如 [384, 768, 768]。
            hidden_dim: 专家模型的隐藏层维度。
            output_dim: 专家模型的输出维度（如分类任务的类别数）。
        """
        super(MultimodalSentimentModel, self).__init__()
        self.feature_dims = [512, 1024, 1024]  # 每个模态的输入维度
        self.shared_dims = [128, 128, 128]  # 模态无关特征维度
        self.private_dims = [128, 128, 128]  # 模态特有特征维度

        # 特征解耦模块
        self.feature_decoupler = FeatureDecoupler(self.feature_dims, self.shared_dims, self.private_dims)

        # 动态蒸馏模块
        self.dynamic_distiller = DynamicGraphDistiller(self.shared_dims, output_dim=128)  # 蒸馏后的输出维度为 128

        # 路由机制
        self.router = RoutingMechanism(sum(self.feature_dims), 128, 1)  # 输入为拼接后的上下文


        # 正式与非正式专家模型
        self.informal_expert = InformalExpert(128, hidden_dim, output_dim)
        self.formal_expert = FormalExpert(128, hidden_dim, output_dim)


    def forward(self, inputs, context):
        """
        前向传播。
        参数:
            inputs: 多模态输入，形状为 List of (batch_size, seq_len, feature_dim)。
            context: 拼接后的上下文，形状为 (batch_size, seq_len, sum(feature_dims))。
        返回:
            模型输出和路由器决策。
        """
        # 1. 特征解耦
        shared_features, private_features = self.feature_decoupler(inputs)

        # 2. 动态蒸馏模态无关特征
        distilled_features = self.dynamic_distiller(shared_features)  # 形状 (batch_size, seq_len, 128)


        # 5. 路由器决策
        route_decision = self.router(context)  # 形状 (batch_size, seq_len, 1)

        # 6. 专家模型预测
        output_formal = self.formal_expert(distilled_features)  # 形状 (batch_size, seq_len, output_dim)
        output_informal = self.informal_expert(distilled_features)  # 形状 (batch_size, seq_len, output_dim)

        # 7. 动态加权融合
        output = route_decision * output_formal + (1 - route_decision) * output_informal

        return output, route_decision, shared_features, private_features, distilled_features



    # def set_requires_grad(self, layer_names, requires_grad):
    #     # 添加冻结/解冻方法，支持逐层解冻
    #     for name, param in self.named_parameters():
    #         if any(layer in name for layer in layer_names):
    #             param.requires_grad = requires_grad

