"""Contains an experiment class for running simulations."""
import datetime
import logging

from GRL_Experiment.Exp_HighwayRamps.registry import make_create_env


class Experiment:

    def __init__(self, flow_params, custom_callables=None):
        """Instantiate the Experiment class.

        Parameters
        ----------
        flow_params : dict
            flow-specific parameters
        custom_callables : dict < str, lambda >
            strings and lambda functions corresponding to some information we
            want to extract from the environment. The lambda will be called at
            each step to extract information from the env and it will be stored
            in a dict keyed by the str.
        """
        self.custom_callables = custom_callables or {}

        # Get the env name and a creator for the environment.
        create_env, _ = make_create_env(flow_params)

        # Create the environment.
        self.env = create_env()

        logging.info(" Starting experiment {} at {}".format(
            self.env.network.name, str(datetime.datetime.utcnow())))

        logging.info("Initializing environment.")

    def run(self, num_HVs, num_AVs, training, testing, Graph):

        import torch
        from GRL_Library.common import replay_buffer, explorer_discrete
        from GRL_Library.agent.Discrete import DistributionalDQN_agent
        from GRL_Utils.Train_and_Test_Q import Training_GRLModels, Testing_GRLModels

        # Initialize GRL model
        N = num_HVs + num_AVs
        F = 2 + self.env.net_params.additional_params['highway_lanes'] \
            + self.env.n_unique_intentions
        A = 3
        n_atoms = 51
        V_min = 0
        V_max = 200
        assert isinstance(Graph, bool)
        if Graph:
            from GRL_Net.Model_Discrete.Q_Distributional import Graph_Model
            GRL_Net = Graph_Model(N, F, A, n_atoms, V_min, V_max)
        else:
            from GRL_Net.Model_Discrete.Q_Distributional import NonGraph_Model
            GRL_Net = NonGraph_Model(N, F, A, n_atoms, V_min, V_max)

        # Initialize optimizer
        optimizer = torch.optim.Adam(GRL_Net.parameters(), eps=0.001)
        # Replay_buffer
        replay_buffer = replay_buffer.ReplayBuffer(size=10 ** 6)
        # Discount factor
        gamma = 0.9
        # Initialize exploration policy
        explorer = explorer_discrete.LinearDecayEpsilonGreedy(start_epsilon=0.5, end_epsilon=0.01, decay_step=5000)

        # for parameters in GRL_Net.parameters():
        #     print("param:", parameters)

        # Initialize GRL agent
        warmup = 10000  # warmup steps
        GRL_DQN = DistributionalDQN_agent.DistributionalDQN(
            GRL_Net,  # model
            optimizer,  # optimizer
            explorer,  # exploration policy
            replay_buffer,  # replay buffer
            gamma,  # discount factor
            batch_size=32,  # batch_size
            warmup_step=warmup,  # warmup steps
            update_interval=10,  # model update interval
            target_update_interval=5000,  # target model update interval
            target_update_method='soft',  # update method
            soft_update_tau=0.1,  # soft_update factor
            n_steps=1,  # multi-steps
            V_min=V_min,  # distribution value interval minimum
            V_max=V_max,  # distribution value interval maximum
            n_atoms=n_atoms,  # number of distribution samples
            model_name="DQN_model"  # model name
        )

        # Training
        n_episodes = 50
        max_episode_len = 2500
        save_dir = '../GRL_TrainedModels/DQN2'
        debug_training = False
        if training:
            Training_GRLModels(GRL_Net, GRL_DQN, self.env, n_episodes, max_episode_len, save_dir, warmup,
                               debug_training)

        # Testing
        test_episodes = 10
        load_dir = 'GRL_TrainedModels/DQN'
        debug_testing = False
        if testing:
            Testing_GRLModels(GRL_Net, GRL_DQN, self.env, test_episodes, load_dir, debug_testing)
