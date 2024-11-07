
import torch
import torch.nn as nn

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

class MultimodalSentimentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MultimodalSentimentModel, self).__init__()
        self.informal_expert = InformalExpert(input_dim, hidden_dim, output_dim)
        self.formal_expert = FormalExpert(input_dim, hidden_dim, output_dim)
        self.router = RoutingMechanism(2560, 128, 1)  # 二元输出用于正式与非正式的区分

    def forward(self, x, context):
        # 使用路由器的输出作为权重，实现软路由的动态加权
        route_decision = self.router(context)
        output_formal = self.formal_expert(x)
        output_informal = self.informal_expert(x)

        output = route_decision * output_formal + (1 - route_decision) * output_informal
        return output, route_decision
