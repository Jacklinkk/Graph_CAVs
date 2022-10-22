# This python file is used to plot the result curve
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


def Mean_and_Std(method, scenario):
    """
        This function is used to calculate the mean as well as the standard deviation
        of the data under different SAMPLES. Three samples are selected.

        Parameters:
        --------
        method: the method used
        scenario: the applicable scenario
    """
    # Data directory retrieval
    gdir_1 = "Data/" + scenario + method + "/graph1"
    gdir_2 = "Data/" + scenario + method + "/graph2"
    gdir_3 = "Data/" + scenario + method + "/graph3"
    ngdir_1 = "Data/" + scenario + method + "/nograph1"
    ngdir_2 = "Data/" + scenario + method + "/nograph2"
    ngdir_3 = "Data/" + scenario + method + "/nograph3"

    # Data loading
    g1 = Data_Loader(gdir_1)
    g2 = Data_Loader(gdir_2)
    g3 = Data_Loader(gdir_3)
    ng1 = Data_Loader(ngdir_1)
    ng2 = Data_Loader(ngdir_2)
    ng3 = Data_Loader(ngdir_3)

    # Data merging
    Data_graph = [g1, g2, g3]
    Data_nograph = [ng1, ng2, ng3]

    # Get the length of the data list
    Length_Data = len(Data_graph)

    Reward_graph = Data_graph
    Reward_nograph = Data_nograph

    # Calculate the mean of each step by column
    Reward_Average_graph = np.average(Reward_graph, axis=0)
    Reward_Average_nograph = np.average(Reward_nograph, axis=0)

    # Calculate the standard deviation of each step by column
    Reward_Std_graph = np.std(Reward_graph, axis=0)
    Reward_Std_nograph = np.std(Reward_nograph, axis=0)

    # Combine data into a matrix
    Reward_Proceed = [Reward_Average_graph, Reward_Std_graph,
                      Reward_Average_nograph, Reward_Std_nograph]

    return Reward_Proceed


def curve_smooth(data, sigma_r, sigma_s):
    """
        This function is used to smooth the curve to plot the data

        Parameters:
        --------
        data: data for the curve
        sigma_r: smoothing parameter for reward data
        sigma_s: smoothing parameter for standard deviation data
    """

    # Reward smoothing
    data[0] = gaussian_filter1d(data[0], sigma=sigma_r)
    data[2] = gaussian_filter1d(data[2], sigma=sigma_r)

    # Standard deviation smoothing
    data[1] = gaussian_filter1d(data[1], sigma=sigma_s)
    data[3] = gaussian_filter1d(data[3], sigma=sigma_s)


def curve_plot(data, algorithm, config, mark, linetype, linecolor):
    """
        This function is used to draw a curve

        Parameters:
        --------
        data: the data for the curve
        algorithm: the algorithm used by the model
        config: configuration of the plotting parameters
        mark: curve marker
        linetype: the line type of the curve
        linecolor: the colour of the curve
    """
    # Get configuration parameters
    linewidth = config[0]
    fontsize = config[1]
    transparency = config[2]
    color = config[3]

    # Naming
    graph_name = algorithm + "_with graph"
    nograph_name = algorithm + "_wo graph"

    # Curve plotting
    ax_Reward.fill_between(x, data[0] + data[1],
                           data[0] - data[1], alpha=transparency, color=color)
    ax_Reward.fill_between(x, data[2] + data[3],
                           data[2] - data[3], alpha=transparency, color=color)
    ax_Reward.plot(x, data[0], color=line_color, linewidth=linewidth, linestyle=linetype[0],
                   label=graph_name,
                   marker=mark[0], markersize=mark[1])
    ax_Reward.plot(x, data[2], color=line_color, linewidth=linewidth, linestyle=linetype[1],
                   label=nograph_name,
                   marker=mark[0], markersize=mark[1])


if __name__ == '__main__':
    # ------ Directory element records ------ #
    # Algorithm arrays
    value_based = ["DQN", "DoubleDQN", "DuelingDQN", "DQN-NoisyNet",
                   "DQN-PER", "DistributionalDQN", "RainbowDQN"]
    policy_based = ["REINFORCE", "AC", "A2C", "NAF", "DoubleNAF",
                    "DDPG", "TD3", "PPO"]
    # Scenarios
    scenario = ["HR/", "FE/"]

    # ------ Data processing (3 samples) ------ #
    # Data reading and processing
    # Highway ramping scenario
    HR_DQN = Mean_and_Std(value_based[0], scenario[0])
    HR_DoubleDQN = Mean_and_Std(value_based[1], scenario[0])
    HR_DuelingDQN = Mean_and_Std(value_based[2], scenario[0])
    HR_DQN_NoisyNet = Mean_and_Std(value_based[3], scenario[0])
    HR_DQN_PER = Mean_and_Std(value_based[4], scenario[0])
    HR_DistributionalDQN = Mean_and_Std(value_based[5], scenario[0])
    HR_RainbowDQN = Mean_and_Std(value_based[6], scenario[0])
    HR_REINFORCE = Mean_and_Std(policy_based[0], scenario[0])
    HR_AC = Mean_and_Std(policy_based[1], scenario[0])
    HR_A2C = Mean_and_Std(policy_based[2], scenario[0])
    HR_PPO = Mean_and_Std(policy_based[7], scenario[0])

    HR_Data = [HR_DQN, HR_DoubleDQN, HR_DuelingDQN, HR_DQN_NoisyNet, HR_DQN_PER,
               HR_DistributionalDQN, HR_RainbowDQN, HR_REINFORCE, HR_AC,
               HR_A2C, HR_PPO]

    # Figure-Eight scenario
    FE_REINFORCE = Mean_and_Std(policy_based[0], scenario[1])
    FE_AC = Mean_and_Std(policy_based[1], scenario[1])
    FE_A2C = Mean_and_Std(policy_based[2], scenario[1])
    FE_NAF = Mean_and_Std(policy_based[3], scenario[1])
    FE_DoubleNAF = Mean_and_Std(policy_based[4], scenario[1])
    FE_DDPG = Mean_and_Std(policy_based[5], scenario[1])
    FE_TD3 = Mean_and_Std(policy_based[6], scenario[1])
    FE_PPO = Mean_and_Std(policy_based[7], scenario[1])

    FE_Data = [FE_REINFORCE, FE_AC, FE_A2C, FE_NAF, FE_DoubleNAF,
               FE_DDPG, FE_TD3, FE_PPO]

    # Calculate and print the average award
    # print("------------------ DQN Reward ------------------")
    # print("DQN_nograph:", np.mean(DQN_Reward_nograph[40:]))
    # print("DQN_graph:", np.mean(DQN_Reward_graph[40:]))
    # print("------------------ Reward ------------------")
    # print("DQN_nograph:", np.mean(DQN_Reward))
    # print("DQN_graph:", np.mean(DQN_Reward_graph))

    # Data further processed
    # Data stored as rewards and standard deviations
    # Data structure as:
    # [graph reward mean, graph standard deviation, nograph reward mean, nograph standard deviation]

    # Data smoothing
    sigma_reward = 2.0
    sigma_std = 10.0
    for i in HR_Data:
        curve_smooth(i, sigma_reward, sigma_std)
    for i in FE_Data:
        curve_smooth(i, sigma_reward, sigma_std)

    # ------ Plotting curve (HR) ------ #
    # Plotting the reward curve
    fig_Reward_HR, ax_Reward = plt.subplots(dpi=240)

    # Defining the horizontal axis (episode)
    length = len(HR_DQN[0])
    x = np.arange(0, length, 1)

    # Plotting curves for different algorithms
    # ------ plotting configuration ------ #
    linewidth = 1.0  # Line width
    fontsize = 10.0  # font size
    transparency = 0.2  # transparency
    color_shadow = "0.6"  # standard deviation shadow colour
    c = [linewidth, fontsize, transparency, color_shadow]

    # ---HR_DQN--- #
    marker = ['', 5]
    linetype = ['-', '--']
    # line_color = (168/255, 3/255, 38/255)
    line_color = 'b'
    curve_plot(HR_A2C, "A2C", c, marker, linetype, line_color)
    # # ---HR_DistributionalDQN--- #
    # marker = ['', 2]
    # linetype = '--'
    # curve_plot(HR_DistributionalDQN, "DDQN", c, marker, linetype, line_color)
    # # ---HR_REINFORCE--- #
    # marker = ['', 2]
    # linetype = ':'
    # curve_plot(HR_REINFORCE, "REINFORCE", c, marker, linetype, line_color)
    # # ---HR_AC--- #
    # marker = ['', 2]
    # linetype = ['-', '--']
    # line_color = (26/255, 40/255, 71/255)
    # curve_plot(HR_AC, "AC", c, marker, linetype, line_color)
    # # ---HR_A2C--- #
    # marker = ['', 2]
    # linetype = ['-', '--']
    # line_color = (65/255, 130/255, 164/255)
    # curve_plot(HR_A2C, "A2C", c, marker, linetype, line_color)

    plt.xticks(size=fontsize)
    plt.yticks(size=fontsize)
    ax_Reward.set_xlabel("Episode", size=fontsize)
    ax_Reward.set_ylabel("Reward", size=fontsize)
    ax_Reward.set_xlim([1, length])
    # ax_Reward.set_ylim([-500, 4000])
    ax_Reward.grid(True)
    ax_Reward.legend(loc='center right', bbox_to_anchor=(1, 0.2), prop={'size': 8})

    # Save the curve
    plt.savefig(fname="Fig/Fig_Training/Reward_HR.jpg", dpi='figure', format="jpg")
    plt.show()

    # --------------------------------- #
    # --------------------------------- #
    # --------------------------------- #

    # ------ Plotting curve (FE) ------
    # Plotting the reward curve
    fig_Reward_FE, ax_Reward = plt.subplots(dpi=240)
    # Define the horizontal axis (episode)
    length = len(FE_AC[0])
    xx = np.arange(0, length, 1)

    # Plotting curves for different algorithms
    # ------ plotting configuration ------ #
    linewidth = 1.0  # Line width
    fontsize = 10.0  # Font size
    transparency = 0.2  # transparency
    color_shadow = "0.6"  # Standard deviation shadow part colour
    c = [linewidth, fontsize, transparency, color_shadow]

    # ---FE_REINFORCE--- #
    marker = ['', 5]
    linetype = ['-', '--']
    # line_color = (168/255, 3/255, 38/255)
    line_color = "b"
    curve_plot(FE_PPO, "PPO", c, marker, linetype, line_color)
    # # ---FE_AC曲--- #
    # marker = ['', 2]
    # linetype = ['-', '--']
    # line_color = (26/255, 40/255, 71/255)
    # curve_plot(FE_PPO, "PPO", c, marker, linetype, line_color)
    # # ---FE_A2C--- #
    # marker = ['', 2]
    # linetype = ['-', '--']
    # line_color = (65/255, 130/255, 164/255)
    # curve_plot(FE_DDPG, "DDPG", c, marker, linetype, line_color)
    # # ---FE_DDPG--- #
    # marker = ['', 2]
    # linetype = ['-', '--']
    # line_color = (168/255, 3/255, 38/255)
    # curve_plot(FE_DDPG, "DDPG", c, marker, linetype, line_color)
    # # ---FE_PPO--- #
    # marker = ['', 2]
    # linetype = ['-', '--']
    # line_color = (168/255, 3/255, 38/255)
    # curve_plot(FE_PPO, "PPO", c, marker, linetype, line_color)

    plt.xticks(size=fontsize)
    plt.yticks(size=fontsize)
    ax_Reward.set_xlabel("Episode", size=fontsize)
    ax_Reward.set_ylabel("Reward", size=fontsize)
    ax_Reward.set_xlim([1, length])
    # ax_Reward.set_ylim([-500, 4000])
    ax_Reward.grid(True)
    ax_Reward.legend(loc='center right', bbox_to_anchor=(1, 0.2), prop={'size': 8})

    # Save curve
    plt.savefig(fname="Fig/Fig_Training/Reward_FE.jpg", dpi='figure', format="jpg")
    plt.show()
