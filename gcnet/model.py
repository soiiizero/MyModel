import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch.distributions import Normal

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

class GaussianDistribution(nn.Module):
    def __init__(self, input_dim):
        super(GaussianDistribution, self).__init__()
        self.mean_layer = nn.Linear(input_dim, input_dim)
        self.log_var_layer = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        mean = self.mean_layer(x)
        log_var = self.log_var_layer(x)
        return mean, log_var

class GATModelWithKL(nn.Module):
    def __init__(self, private_dims, input_dim, hidden_dim, output_dim, num_heads=4, kl_threshold=0.5):
        super(GATModelWithKL, self).__init__()
        # 对私有特征进行高斯分布建模
        self.gaussian = nn.ModuleList([
            GaussianDistribution(private_dim) for private_dim in private_dims
        ])
        # 使用多头注意力机制
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=0.3)
        self.gat2 = GATConv(hidden_dim * num_heads, output_dim, heads=1, dropout=0.3)
        self.kl_threshold = kl_threshold

    @staticmethod
    def kl_divergence(mean_p, log_var_p, mean_q, log_var_q):
        """计算两个高斯分布之间的KL散度"""
        var_p = torch.exp(log_var_p)
        var_q = torch.exp(log_var_q)
        kl_loss = 0.5 * (log_var_q - log_var_p +
                         (var_p + (mean_p - mean_q).pow(2)) / var_q - 1)
        return torch.sum(kl_loss, dim=-1)

    def construct_graph(self, private_means, private_log_vars, batch_size):
        """为每个batch构建图结构"""
        batch_edges = []
        device = private_means[0].device  # 获取设备信息
        for b in range(batch_size):
            # 获取当前batch的分布参数
            batch_means = [means[b] for means in private_means]
            batch_log_vars = [log_vars[b] for log_vars in private_log_vars]

            # 计算所有模态对之间的KL散度
            for i in range(len(batch_means)):
                for j in range(i + 1, len(batch_means)):
                    kl_ij = self.kl_divergence(
                        batch_means[i], batch_log_vars[i],
                        batch_means[j], batch_log_vars[j]
                    )

                    if torch.mean(kl_ij) > self.kl_threshold:
                        # 为当前batch添加边，需要考虑batch中的节点偏移
                        offset = b * len(private_means)
                        batch_edges.extend([
                            [i + offset, j + offset],
                            [j + offset, i + offset]  # 添加双向边
                        ])

        if not batch_edges:  # 如果没有边，添加自环
            batch_edges = [[i, i] for i in range(batch_size * len(private_means))]

        return torch.tensor(batch_edges, dtype=torch.long,device=device).t()

    def forward(self, distilled_features, private_features):
        """
        参数:
            distilled_features: [batch_size, seq_len, input_dim]
            private_features: list of [batch_size, seq_len, private_dim]
        返回:
            outputs: [batch_size, seq_len, output_dim]
        """
        batch_size = distilled_features.shape[0]
        device = distilled_features.device  # 获取输入设备

        # 确保模型在正确的设备上
        self.to(device)
        # 1. 对私有特征进行分布建模
        private_means = []
        private_log_vars = []
        for i, features in enumerate(private_features):
            features = features.to(device)  # 确保特征在正确的设备上
            mean, log_var = self.gaussian[i](features)
            private_means.append(mean)
            private_log_vars.append(log_var)

        # 2. 构建图结构
        edges = self.construct_graph(private_means, private_log_vars, batch_size)

        # 3. 重塑特征以适应PyG的输入格式
        # 将batch和seq维度展平
        x = distilled_features.reshape(-1, distilled_features.shape[-1])

        # 4. 应用GAT层
        x = self.gat1(x, edges)
        x = torch.relu(x)
        x = self.gat2(x, edges)

        # 5. 重塑回原始维度
        outputs = x.reshape(batch_size, -1, x.shape[-1])

        return outputs

# class RoutingMechanism(nn.Module):
#     def __init__(self, input_dim=2560, hidden_dim=128, output_dim=1, num_heads=4, dropout_rate=0.3):
#         super(RoutingMechanism, self).__init__()
#         # 增加多头注意力层
#         self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout_rate)
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout_rate)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, context):
#         context, _ = self.attention(context, context, context)
#         context = self.fc1(context)
#         context = self.relu(context)
#         context = self.dropout(context)
#         context = self.fc2(context)
#         return torch.sigmoid(context)  # 输出在0到1之间，用于动态加权

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

class SceneConfidencePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SceneConfidencePredictor, self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.predictor(x)

class SceneAdaptiveRouter(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SceneAdaptiveRouter, self).__init__()
        # 内部置信度预测器 - 类似Mono-Confidence
        self.formal_confidence = SceneConfidencePredictor(input_dim, hidden_dim)
        self.informal_confidence = SceneConfidencePredictor(input_dim, hidden_dim)

        # 场景分类器 - 用于计算场景分布
        self.scene_classifier = nn.Linear(input_dim, 2)

    def compute_distribution_uniformity(self, logits):
        probs = F.softmax(logits, dim=-1)
        mean_prob = 1.0 / probs.size(-1)
        return torch.mean(torch.abs(probs - mean_prob), dim=-1, keepdim=True)

    def forward(self, context):
        # 1. 计算场景内部置信度
        formal_conf = self.formal_confidence(context)
        informal_conf = self.informal_confidence(context)

        # 2. 计算场景分布一致性
        scene_logits = self.scene_classifier(context)
        formal_du = self.compute_distribution_uniformity(scene_logits)
        informal_du = 1 - formal_du  # 简化处理，两个场景分布一致性互补

        # 3. 计算场景置信度 (Co-Belief)
        formal_belief = formal_conf
        informal_belief = informal_conf

        # 4. 相对校准
        scene_rc = formal_du / (informal_du + 1e-8)
        k_formal = torch.where(
            formal_du < informal_du,
            scene_rc,
            torch.ones_like(scene_rc)
        )
        k_informal = torch.where(
            informal_du < formal_du,
            1 / scene_rc,
            torch.ones_like(scene_rc)
        )

        # 5. 最终的场景权重
        formal_weight = formal_belief * k_formal
        informal_weight = informal_belief * k_informal

        # 6. 归一化权重
        total_weight = formal_weight + informal_weight
        formal_weight = formal_weight / total_weight

        return formal_weight, {
            'formal_conf': formal_conf,
            'informal_conf': informal_conf,
            'formal_du': formal_du,
            'informal_du': informal_du,
            'formal_weight': formal_weight,
            'informal_weight': informal_weight
        }

class MultimodalSentimentModel(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        """
        多模态情感分析模型。
        参数:
            feature_dims: 每个模态的输入维度列表，如 [512, 1024, 1024]。
            shared_dims: 每个模态的无关特征维度列表，如 [128, 128, 128]。
            private_dims: 每个模态的特有特征维度列表，如 [128, 128, 128]。
            hidden_dim: 专家模型的隐藏层维度。
            output_dim: 专家模型的输出维度（如分类任务的类别数）。
        """
        super(MultimodalSentimentModel, self).__init__()
        self.feature_dims = [512, 1024, 1024]  # 每个模态的输入维度
        self.shared_dims = [128, 128, 128]  # 模态无关特征维度
        self.private_dims = [128, 128, 128]  # 模态特有特征维度

        # 特征解耦模块
        self.feature_decoupler = FeatureDecoupler(self.feature_dims, self.shared_dims, self.private_dims)

        # 动态融合模块
        self.dynamic_distiller = DynamicGraphDistiller(self.shared_dims, output_dim=128)  # 蒸馏后的输出维度为 128

        # 图卷积模型
        self.gat_model = GATModelWithKL(self.private_dims, input_dim=128, hidden_dim=hidden_dim, output_dim=128, num_heads=4)

        # 替换原来的router
        self.scene_router = SceneAdaptiveRouter(sum(self.feature_dims), hidden_dim)

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

        # 融合模态无关特征
        distilled_features = self.dynamic_distiller(shared_features)

        # 图注意力卷积
        output = self.gat_model(distilled_features, private_features)

        # 场景自适应路由
        route_weight, confidences = self.scene_router(context)
        # 5. 专家模型预测
        output_formal = self.formal_expert(output)  # 形状 (batch_size, seq_len, output_dim)
        output_informal = self.informal_expert(output)  # 形状 (batch_size, seq_len, output_dim)

        # 基于置信度的动态融合
        output = route_weight * output_formal + (1 - route_weight) * output_informal

        return output, route_weight, shared_features, private_features, distilled_features


