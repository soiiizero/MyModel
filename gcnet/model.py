
import torch
import torch.nn as nn

class ExpertModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ExpertModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

class RoutingMechanism(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=1):
        super(RoutingMechanism, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)  # Output between 0 and 1 for routing decision

class MultimodalSentimentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MultimodalSentimentModel, self).__init__()
        self.informal_expert = ExpertModel(input_dim, hidden_dim, output_dim)
        self.formal_expert = ExpertModel(input_dim, hidden_dim, output_dim)
        self.router = RoutingMechanism(256, 128, 1)  # Output 1 value for binary decision

    def forward(self, x, context):

        # Use router to determine which expert to use based on the context (formal/informal)
        route_decision = self.router(context)
        if route_decision >= 0.5:
            return self.formal_expert(x)
        else:
            return self.informal_expert(x)
