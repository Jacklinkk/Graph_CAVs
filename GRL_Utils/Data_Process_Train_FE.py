import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


def Data_Loader(data_dir):
    """
        This function is used to read the saved data

        Parameters:
        --------
        data_dir: the directory where the model and data are stored
    """

    # Get the directory
    Reward_dir = data_dir + "/Rewards.npy"

    # Read data via numpy
    Reward = np.load(Reward_dir)

    return Reward


def Mean_and_Std(Data):
    """
        This function is used to calculate the mean and standard deviation of the data under different samples

        Parameter description:
        --------
        Data: The list of data to be calculated.
    """

    # Get the length of the data list
    Length_Data = len(Data)

    # Calculate the mean and standard deviation of each indicator
    # -------------------------------------------------------------- #
    # 1.Reward processing
    Reward = []
    for i in range(0, Length_Data):
        Reward.append(Data[i][0])
    Reward_Average = np.average(Reward, axis=0)
    Reward_Std = np.std(Reward, axis=0)
    Reward_Proceed = [Reward_Average, Reward_Std]
    # -------------------------------------------------------------- #

    return Reward_Proceed


if __name__ == '__main__':
    # 1.Catalogue input (3 random samples)
    value_based = ["NAF", "DoubleNAF"]

    policy_based = ["REINFORCE", "AC", "A2C", "PPO", "DDPG", "TD3"]

    method = "PPO"

    graph_1 = "Data/FE/" + method + "/graph1"
    graph_2 = "Data/FE/" + method + "/graph2"
    graph_3 = "Data/FE/" + method + "/graph3"

    nograph_1 = "Data/FE/" + method + "/nograph1"
    nograph_2 = "Data/FE/" + method + "/nograph2"
    nograph_3 = "Data/FE/" + method + "/nograph3"

    # 2.Data loading
    Data_graph_1 = Data_Loader(graph_1)
    Data_graph_2 = Data_Loader(graph_2)
    Data_graph_3 = Data_Loader(graph_3)

    Data_nograph_1 = Data_Loader(nograph_1)
    Data_nograph_2 = Data_Loader(nograph_2)
    Data_nograph_3 = Data_Loader(nograph_3)

    # 3.Data selection
    if method in value_based:
        Data_graph = [Data_graph_1[10:], Data_graph_2[10:], Data_graph_3[10:]]
        Data_nograph = [Data_nograph_1[10:], Data_nograph_2[10:], Data_nograph_3[10:]]
    elif method in policy_based:
        Data_graph = [Data_graph_1, Data_graph_2, Data_graph_3]
        Data_nograph = [Data_nograph_1, Data_nograph_2, Data_nograph_3]
    else:
        raise ValueError

    # 4.Mean value
    Mean_graph = np.mean(Data_graph)
    Mean_nograph = np.mean(Data_nograph)

    # 5.Std value
    Std_graph = np.std(Data_graph)
    Std_nograph = np.std(Data_nograph)

    print(method + ":")
    # 6.Print mean reward
    print("------------------ Reward ------------------")
    print("Graph_Reward: %.2f" % Mean_graph)
    print("Nograph_Reward: %.2f" % Mean_nograph)
    OP = 100 * (Mean_graph - Mean_nograph) / Mean_nograph
    print("Optimization Rate: %.2f" % OP)

    # 7.Print std value
    print("------------------ Std ------------------")
    print("Graph_Std: %.2f" % Std_graph)
    print("Nograph_Std: %.2f" % Std_nograph)

