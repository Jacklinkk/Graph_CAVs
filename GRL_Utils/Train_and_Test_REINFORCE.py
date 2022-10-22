# The python file includes the training and testing functions for the GRL model
import numpy as np


def Training_GRLModels(GRL_Net, GRL_model, env, n_episodes, max_episode_len, save_dir, debug):
    """
        This function is a training function for the GRL model

        Parameters:
        --------
        GRL_Net: the neural network used in the GRL model
        GRL_model: the GRL model to be trained
        env: the simulation environment registered to gym
        n_episodes: number of training rounds
        max_episode_len: the maximum number of steps to train in a single step
        save_dir: path to save the model
        debug: debug-related model parameters
    """
    # The following is the model training process
    Rewards = []  # Initialize the reward matrix for data saving
    Loss = []  # Initialize the Loss matrix for data saving
    Episode_Steps = []  # Initialize the step matrix to hold the step length at task completion for each episode

    print("#------------------------------------#")
    print("#----------Training Begins-----------#")
    print("#------------------------------------#")

    for i in range(1, n_episodes + 1):
        if debug:
            print("#------------------------------------#")
            for parameters in GRL_Net.parameters():
                print("param:", parameters)
            print("#------------------------------------#")
        obs = env.reset()
        R = 0
        t = 0
        while True:
            # ------Action generation------ #
            action = GRL_model.choose_action(obs)  # The agent interacts with the environment

            obs_next, reward, done, info = env.step(action)
            R += reward
            t += 1
            reset = t == max_episode_len

            # ------Save reward------ #
            GRL_model.store_rewards(reward)

            # ------Observation update------ #
            obs = obs_next

            if done or reset:
                break

        # ------ Records training data ------ #
        # Get the training data
        training_data = GRL_model.get_statistics()
        loss = training_data[0]
        # Record the training data
        Rewards.append(R)
        Episode_Steps.append(t)
        Loss.append(loss)

        # ------Policy update------ #
        GRL_model.learn()

        # ------Print the training data------ #
        if i % 1 == 0:
            print('Training Episode:', i, 'Reward:', R, 'Loss:', loss)
    print('Training Finished.')

    # Save model
    GRL_model.save_model(save_dir)
    # Save other data
    np.save(save_dir + "/Rewards", Rewards)
    np.save(save_dir + "/Episode_Steps", Episode_Steps)
    np.save(save_dir + "/Loss", Loss)


def Testing_GRLModels(GRL_Net, GRL_model, env, test_episodes, load_dir, debug):
    """
        This function is a test function for a trained GRL model

        Parameters:
        --------
        GRL_Net: the neural network used in the GRL model
        GRL_model: the GRL model to be tested
        env: the simulation environment registered to gym
        test_episodes: the number of rounds to be tested
        load_dir: path to read the model
        debug: debug-related model parameters
    """
    # Here is how the model is tested
    Rewards = []  # Initialize the reward matrix for data storage

    GRL_model.load_model(load_dir)

    print("#-------------------------------------#")
    print("#-----------Testing Begins------------#")
    print("#-------------------------------------#")

    for i in range(1, test_episodes + 1):
        if debug:
            print("#------------------------------------#")
            for parameters in GRL_Net.parameters():
                print("param:", parameters)
            print("#------------------------------------#")
        obs = env.reset()
        R = 0
        t = 0
        while True:
            action = GRL_model.choose_action(obs)
            obs, reward, done, info = env.step(action)
            R += reward
            t += 1
            reset = done

            if done or reset:
                break

        Rewards.append(R)
        print('Evaluation Episode:', i, 'Reward:', R)
    print('Evaluation Finished')

    np.save(load_dir + "/Test_Rewards", Rewards)
