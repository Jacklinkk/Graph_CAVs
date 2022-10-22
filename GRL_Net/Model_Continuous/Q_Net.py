import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse


def datatype_transmission(states, device):
    """
        1.This function is used to convert observations in the environment to the
        float32 Tensor data type that pytorch can accept.
        2.Pay attention: Depending on the characteristics of the data structure of
        the observation, the function needs to be changed accordingly.
    """
    features = torch.as_tensor(states[0], dtype=torch.float32, device=device)
    adjacency = torch.as_tensor(states[1], dtype=torch.float32, device=device)
    mask = torch.as_tensor(states[2], dtype=torch.float32, device=device)

    return features, adjacency, mask


# ------Graph Model------ #
class Graph_Model(nn.Module):
    """
        1.N is the number of vehicles
        2.F is the feature length of each vehicle
        3.A is the number of selectable actions
    """
    def __init__(self, N, F, A, action_min, action_max):
        super(Graph_Model, self).__init__()
        self.num_agents = N
        self.num_outputs = A
        self.action_min = action_min
        self.action_max = action_max

        # Encoder
        self.encoder_1 = nn.Linear(F, 32)
        self.encoder_2 = nn.Linear(32, 32)

        # GNN
        self.GraphConv = GCNConv(32, 32)
        self.GraphConv_Dense = nn.Linear(32, 32)

        # Policy network
        self.policy_1 = nn.Linear(64, 32)
        self.policy_2 = nn.Linear(32, 32)

        # Normalized Advantage Functions (NAF)
        self.value = nn.Linear(32, 1)
        self.mu = nn.Linear(32, A)
        dim_L0 = int(self.num_outputs * (self.num_outputs + 1) / 2)
        self.L0 = nn.Linear(32, dim_L0)

        # GPU configuration
        if torch.cuda.is_available():
            GPU_num = torch.cuda.current_device()
            self.device = torch.device("cuda:{}".format(GPU_num))
        else:
            self.device = "cpu"

        self.to(self.device)

    def forward(self, observation, action=None):
        """
            1.The data type here is numpy.ndarray, which needs to be converted to a
            Tensor data type.
            2.Observation is the state observation matrix, including X_in, A_in_Dense
            and RL_indice.
            3.X_in is the node feature matrix, A_in_Dense is the dense adjacency matrix
            (NxN) (original input)
            4.A_in_Sparse is the sparse adjacency matrix COO (2xnum), RL_indice is the
            reinforcement learning index of controlled vehicles.
        """

        X_in, A_in_Dense, RL_indice = datatype_transmission(observation, self.device)

        # Encoder
        X = self.encoder_1(X_in)
        X = F.relu(X)
        X = self.encoder_2(X)
        X = F.relu(X)

        # GCN
        A_in_Sparse, _ = dense_to_sparse(A_in_Dense)
        X_graph = self.GraphConv(X, A_in_Sparse)
        X_graph = F.relu(X_graph)
        X_graph = self.GraphConv_Dense(X_graph)
        X_graph = F.relu(X_graph)

        # Features concatenation
        F_concat = torch.cat((X_graph, X), 1)

        # Policy
        X_policy = self.policy_1(F_concat)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_2(X_policy)
        X_policy = F.relu(X_policy)

        # ------NAF operation------ #
        # value and mu
        value = self.value(X_policy)
        mu = torch.tanh(self.mu(X_policy))
        mu = mu.unsqueeze(-1)

        # Calculate the lower triangular matrix L
        L0 = torch.tanh(self.L0(X_policy))
        L = torch.zeros(self.num_agents, self.num_outputs, self.num_outputs).to(self.device)
        # get lower triagular indices
        tril_indices = torch.tril_indices(row=self.num_outputs, col=self.num_outputs, offset=0)
        # fill matrix with entries
        L[:, tril_indices[0], tril_indices[1]] = L0
        L.diagonal(dim1=1, dim2=2).exp_()
        # calculate state-dependent, positive-definite square matrix
        P = L * L.transpose(2, 1)

        # Advantage and Q value
        Q = None
        if action is not None:
            action = action.unsqueeze(-1).unsqueeze(-1)
            a = action - mu
            A = (-0.5 * torch.matmul(torch.matmul(a.transpose(2, 1), P), a)).squeeze(-1)
            Q = A + value

        # Action generation
        # add noise to action mu:
        dist = MultivariateNormal(mu.squeeze(-1), torch.inverse(P))
        # dist = Normal(action_value.squeeze(-1), 1)
        action = dist.sample()
        action = torch.clamp(action, min=self.action_min, max=self.action_max)
        # Mask
        mask = torch.reshape(RL_indice, (self.num_agents, 1))
        # Output calculation
        action = torch.mul(action, mask)

        # Store action-related information in the dictionary
        action = action.squeeze(-1)
        action_dict = {'action': action, 'action_min': self.action_min, 'action_max': self.action_max}

        return action_dict, Q, value


# ------NonGraph Model------ #
class NonGraph_Model(nn.Module):
    """
        1.N is the number of vehicles
        2.F is the feature length of each vehicle
        3.A is the number of selectable actions
    """
    def __init__(self, N, F, A, action_min, action_max):
        super(NonGraph_Model, self).__init__()
        self.num_agents = N
        self.num_outputs = A
        self.action_min = action_min
        self.action_max = action_max

        # Policy network
        self.policy_1 = nn.Linear(F, 32)
        self.policy_2 = nn.Linear(32, 32)

        # Normalized Advantage Functions (NAF)
        self.value = nn.Linear(32, 1)
        self.mu = nn.Linear(32, A)
        dim_L0 = int(self.num_outputs * (self.num_outputs + 1) / 2)
        self.L0 = nn.Linear(32, dim_L0)

        # GPU configuration
        if torch.cuda.is_available():
            GPU_num = torch.cuda.current_device()
            self.device = torch.device("cuda:{}".format(GPU_num))
        else:
            self.device = "cpu"

        self.to(self.device)

    def forward(self, observation, action=None):
        """
            1.The data type here is numpy.ndarray, which needs to be converted to a
            Tensor data type.
            2.Observation is the state observation matrix, including X_in, and RL_indice.
            3.X_in is the node feature matrix, RL_indice is the reinforcement learning
            index of controlled vehicles.
        """

        X_in, _, RL_indice = datatype_transmission(observation, self.device)

        # Policy
        X_policy = self.policy_1(X_in)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_2(X_policy)
        X_policy = F.relu(X_policy)

        # ------NAF operation------ #
        # Value and mu calculation
        value = self.value(X_policy)
        mu = torch.tanh(self.mu(X_policy))
        mu = mu.unsqueeze(-1)

        # Calculate the lower triangular matrix L
        L0 = torch.tanh(self.L0(X_policy))
        L = torch.zeros(self.num_agents, self.num_outputs, self.num_outputs).to(self.device)
        # get lower triagular indices
        tril_indices = torch.tril_indices(row=self.num_outputs, col=self.num_outputs, offset=0)
        # fill matrix with entries
        L[:, tril_indices[0], tril_indices[1]] = L0
        L.diagonal(dim1=1, dim2=2).exp_()
        # calculate state-dependent, positive-definite square matrix
        P = L * L.transpose(2, 1)

        # Advantage and Q value
        Q = None
        if action is not None:
            action = action.unsqueeze(-1).unsqueeze(-1)
            a = action - mu
            A = (-0.5 * torch.matmul(torch.matmul(a.transpose(2, 1), P), a)).squeeze(-1)
            Q = A + value

        # Action generation
        # add noise to action mu:
        dist = MultivariateNormal(mu.squeeze(-1), torch.inverse(P))
        # dist = Normal(action_value.squeeze(-1), 1)
        action = dist.sample()
        action = torch.clamp(action, min=self.action_min, max=self.action_max)
        # Mask
        mask = torch.reshape(RL_indice, (self.num_agents, 1))
        # Output calculation
        action = torch.mul(action, mask)

        # Store action-related information in the dictionary
        action = action.squeeze(-1)
        action_dict = {'action': action, 'action_min': self.action_min, 'action_max': self.action_max}

        return action_dict, Q, value