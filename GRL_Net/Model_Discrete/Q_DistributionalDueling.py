import torch
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


# ------Graph Model------ #
class Graph_Model(nn.Module):
    """
        1.N is the number of vehicles
        2.F is the feature length of each vehicle
        3.A is the number of selectable actions
        4.n_atoms is the number of distribution samples
        5.V_min is the minimum value of the distribution value interval
        6.V_max is the maximum value of the distribution value interval
    """
    def __init__(self, N, F, A, n_atoms, V_min, V_max):
        super(Graph_Model, self).__init__()
        self.num_agents = N
        self.num_outputs = A
        self.n_atoms = n_atoms
        self.V_min = V_min
        self.V_max = V_max

        # Encoder
        self.encoder_1 = nn.Linear(F, 32)
        self.encoder_2 = nn.Linear(32, 32)

        # GNN
        self.GraphConv = GCNConv(32, 32)
        self.GraphConv_Dense = nn.Linear(32, 32)

        # Policy Network
        self.policy_1 = nn.Linear(64, 32)
        self.policy_2 = nn.Linear(32, 32)

        # Dueling network
        self.policy_advantage = nn.Linear(32, A * n_atoms)
        self.policy_value = nn.Linear(32, n_atoms)

        # GPU configuration
        if torch.cuda.is_available():
            GPU_num = torch.cuda.current_device()
            self.device = torch.device("cuda:{}".format(GPU_num))
        else:
            self.device = "cpu"

        self.to(self.device)

    def dist(self, observation):
        """
            dist finds the q-value distribution for observations
            and provides input for subsequent calls to the forward operation.

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

        # Mask
        mask = torch.reshape(RL_indice, (self.num_agents, 1))

        # Dueling operation
        Value = self.policy_value(X_policy)
        Value = torch.mul(Value, mask)
        Value = Value.view(-1, 1, self.n_atoms)

        Advantage = self.policy_advantage(X_policy)
        Advantage = torch.mul(Advantage, mask)
        Advantage = Advantage.view(-1, self.num_outputs, self.n_atoms)

        # ------Distributed operation------ #
        # q_atom calculation
        q_atom = Value + Advantage - Advantage.mean(dim=1, keepdim=True)
        # softmax
        q_distribution = F.softmax(q_atom, dim=-1)
        q_distribution = q_distribution.clamp(min=1e-3)  # 防止nan数据的生成

        return q_distribution

    def forward(self, observation):
        """
            Forward propagation for q_distribution for q tables
        """
        q_distribution = self.dist(observation)

        # Support calculation
        support = torch.linspace(self.V_min, self.V_max, self.n_atoms).to(self.device)
        # Q value calculations
        q_value = torch.sum(q_distribution * support, dim=-1)

        return q_value


# ------NonGraph Model------ #
class NonGraph_Model(nn.Module):
    """
        1.N is the number of vehicles
        2.F is the feature length of each vehicle
        3.A is the number of selectable actions
        4.n_atoms is the number of distribution samples
        5.V_min is the minimum value of the distribution value interval
        6.V_max is the maximum value of the distribution value interval
    """
    def __init__(self, N, F, A, n_atoms, V_min, V_max):
        super(NonGraph_Model, self).__init__()
        self.num_agents = N
        self.num_outputs = A
        self.n_atoms = n_atoms
        self.V_min = V_min
        self.V_max = V_max

        # Policy network
        self.policy_1 = nn.Linear(F, 32)
        self.policy_2 = nn.Linear(32, 32)

        # Dueling network
        self.policy_advantage = nn.Linear(32, A * n_atoms)
        self.policy_value = nn.Linear(32, n_atoms)

        # GPU configuration
        if torch.cuda.is_available():
            GPU_num = torch.cuda.current_device()
            self.device = torch.device("cuda:{}".format(GPU_num))
        else:
            self.device = "cpu"

        self.to(self.device)

    def dist(self, observation):
        """
            dist finds the q-value distribution for observations
            and provides input for subsequent calls to the forward operation.

            1.The data type here is numpy.ndarray, which needs to be converted to a
            Tensor data type.
            2.Observation is the state observation matrix, including X_in, A_in_Dense
            and RL_indice.
            3.X_in is the node feature matrix, A_in_Dense is the dense adjacency matrix
            (NxN) (original input)
            4.A_in_Sparse is the sparse adjacency matrix COO (2xnum), RL_indice is the
            reinforcement learning index of controlled vehicles.
        """

        X_in, _, RL_indice = datatype_transmission(observation, self.device)

        # Policy
        X_policy = self.policy_1(X_in)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_2(X_policy)
        X_policy = F.relu(X_policy)

        # Mask
        mask = torch.reshape(RL_indice, (self.num_agents, 1))

        # Dueling operation
        Value = self.policy_value(X_policy)
        Value = torch.mul(Value, mask)
        Value = Value.view(-1, 1, self.n_atoms)

        Advantage = self.policy_advantage(X_policy)
        Advantage = torch.mul(Advantage, mask)
        Advantage = Advantage.view(-1, self.num_outputs, self.n_atoms)

        # ------Distributed operation------ #
        # q_atom calculation
        q_atom = Value + Advantage - Advantage.mean(dim=1, keepdim=True)
        # softmax
        q_distribution = F.softmax(q_atom, dim=-1)
        q_distribution = q_distribution.clamp(min=1e-3)

        return q_distribution

    def forward(self, observation):
        """
            Forward propagation for q_distribution for q tables
        """
        q_distribution = self.dist(observation)

        # Support calculation
        support = torch.linspace(self.V_min, self.V_max, self.n_atoms).to(self.device)
        # Q value calculation
        q_value = torch.sum(q_distribution * support, dim=-1)

        return q_value
