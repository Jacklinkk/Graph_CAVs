import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
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


# ------Graph Actor Model------ #
class Graph_Actor_Model(nn.Module):
    """
        1.N is the number of vehicles
        2.F is the feature length of each vehicle
        3.A is the number of selectable actions
    """
    def __init__(self, N, F, A, lr, action_min, action_max):
        super(Graph_Actor_Model, self).__init__()
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

        # Actor network
        self.mu = nn.Linear(32, A)
        self.sigma = nn.Linear(32, A)

        # GPU configuration
        if torch.cuda.is_available():
            GPU_num = torch.cuda.current_device()
            self.device = torch.device("cuda:{}".format(GPU_num))
        else:
            self.device = "cpu"

        self.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, observation):
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
        A_in_Sparse, _ = dense_to_sparse(A_in_Dense)  # 将observation的邻接矩阵转换成稀疏矩阵
        X_graph = self.GraphConv(X, A_in_Sparse)
        X_graph = F.relu(X_graph)
        X_graph = self.GraphConv_Dense(X_graph)
        X_graph = F.relu(X_graph)

        # Feature concatenation
        F_concat = torch.cat((X_graph, X), 1)

        # Policy
        X_policy = self.policy_1(F_concat)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_2(X_policy)
        X_policy = F.relu(X_policy)

        # mu and sigma
        pi_mu = self.mu(X_policy)
        pi_sigma = self.sigma(X_policy)

        # Action and log value
        pi_sigma = torch.exp(pi_sigma)
        action_probabilities = torch.distributions.Normal(pi_mu, pi_sigma)
        action = action_probabilities.sample()
        log_probs = action_probabilities.log_prob(action)
        # Action limitation
        action = torch.clamp(action, min=self.action_min, max=self.action_max)

        return action, log_probs, action_probabilities


# ------Graph Critic Model------ #
class Graph_Critic_Model(nn.Module):
    """
        1.N is the number of vehicles
        2.F is the feature length of each vehicle
        3.A is the number of selectable actions
    """
    def __init__(self, N, F, A, lr, action_min, action_max):
        super(Graph_Critic_Model, self).__init__()
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

        # Critic network
        self.value = nn.Linear(32, 1)

        # GPU configuration
        if torch.cuda.is_available():
            GPU_num = torch.cuda.current_device()
            self.device = torch.device("cuda:{}".format(GPU_num))
        else:
            self.device = "cpu"

        self.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, observation):
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

        # Feature concatenation
        F_concat = torch.cat((X_graph, X), 1)

        # Policy
        X_policy = self.policy_1(F_concat)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_2(X_policy)
        X_policy = F.relu(X_policy)

        # Mask
        mask = torch.reshape(RL_indice, (self.num_agents, 1))

        # Value
        value = self.value(X_policy)
        value = torch.mul(value, mask)

        return value


# ------NonGraph Actor Model------ #
class NonGraph_Actor_Model(nn.Module):
    """
        1.N is the number of vehicles
        2.F is the feature length of each vehicle
        3.A is the number of selectable actions
    """
    def __init__(self, N, F, A, lr, action_min, action_max):
        super(NonGraph_Actor_Model, self).__init__()
        self.num_agents = N
        self.num_outputs = A
        self.action_min = action_min
        self.action_max = action_max

        # Policy network
        self.policy_1 = nn.Linear(F, 32)
        self.policy_2 = nn.Linear(32, 32)

        # Actor network
        self.mu = nn.Linear(32, A)
        self.sigma = nn.Linear(32, A)

        # GPU configuration
        if torch.cuda.is_available():
            GPU_num = torch.cuda.current_device()
            self.device = torch.device("cuda:{}".format(GPU_num))
        else:
            self.device = "cpu"

        self.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, observation):
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

        # mu and sigma
        pi_mu = self.mu(X_policy)
        pi_sigma = self.sigma(X_policy)

        # Action and log value
        pi_sigma = torch.exp(pi_sigma)
        action_probabilities = torch.distributions.Normal(pi_mu, pi_sigma)
        action = action_probabilities.sample()
        log_probs = action_probabilities.log_prob(action)
        # Action limitation
        action = torch.clamp(action, min=self.action_min, max=self.action_max)

        return action, log_probs, action_probabilities


# ------NonGraph Critic Model------ #
class NonGraph_Critic_Model(nn.Module):
    """
        1.N is the number of vehicles
        2.F is the feature length of each vehicle
        3.A is the number of selectable actions
    """
    def __init__(self, N, F, A, lr, action_min, action_max):
        super(NonGraph_Critic_Model, self).__init__()
        self.num_agents = N
        self.num_outputs = A
        self.action_min = action_min
        self.action_max = action_max

        # Policy network
        self.policy_1 = nn.Linear(F, 32)
        self.policy_2 = nn.Linear(32, 32)

        # Critic network
        self.value = nn.Linear(32, 1)

        # GPU configuration
        if torch.cuda.is_available():
            GPU_num = torch.cuda.current_device()
            self.device = torch.device("cuda:{}".format(GPU_num))
        else:
            self.device = "cpu"

        self.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, observation):
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

        # Mask
        mask = torch.reshape(RL_indice, (self.num_agents, 1))

        # Value
        value = self.value(X_policy)
        value = torch.mul(value, mask)

        return value
