"""Environment for training the acceleration behavior of vehicles in a ring."""
import torch
from flow.core import rewards
from GRL_Envs.FigureEight.FE_base import Env
# from flow.envs.base import Env

from gym.spaces.box import Box
from gym.spaces import Tuple

import numpy as np
from sklearn.metrics import euclidean_distances

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    'max_accel': 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    'max_decel': 3,
    # desired velocity for all vehicles in the network, in m/s
    'target_velocity': 40,
    # specifies whether vehicles are to be sorted by position during a
    # simulation step. If set to True, the environment parameter
    # self.sorted_ids will return a list of all vehicles sorted in accordance
    # with the environment
    'sort_vehicles': False
}


class AccelEnv(Env):
    """Fully observed acceleration environment.

    This environment used to train autonomous vehicles to improve traffic flows
    when acceleration actions are permitted by the rl agent.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * target_velocity: desired velocity for all vehicles in the network, in m/s
    * sort_vehicles: specifies whether vehicles are to be sorted by position
      during a simulation step. If set to True, the environment parameter
      self.sorted_ids will return a list of all vehicles sorted in accordance
      with the environment

    States
        The state consists of the velocities and absolute position of all
        vehicles in the network. This assumes a constant number of vehicles.

    Actions
        Actions are a list of acceleration for each rl vehicles, bounded by the
        maximum accelerations and decelerations specified in EnvParams.

    Rewards
        The reward function is the two-norm of the distance of the speed of the
        vehicles in the network from the "target_velocity" term. For a
        description of the reward, see: flow.core.rewards.desired_speed

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.

    Attributes
    ----------
    prev_pos : dict
        dictionary keeping track of each veh_id's previous position
    absolute_position : dict
        dictionary keeping track of each veh_id's absolute position
    obs_var_labels : list of str
        referenced in the visualizer. Tells the visualizer which
        metrics to track
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter \'{}\' not supplied'.format(p))

        # variables used to sort vehicles by their initial position plus
        # distance traveled
        self.prev_pos = dict()
        self.absolute_position = dict()

        super().__init__(env_params, sim_params, network, simulator)

    @property
    def action_space(self):
        """See class definition."""
        # return Box(
        #     low=-abs(self.env_params.additional_params['max_decel']),
        #     high=self.env_params.additional_params['max_accel'],
        #     shape=(self.initial_vehicles.num_rl_vehicles, ),
        #     dtype=np.float32)
        N = self.initial_vehicles.num_vehicles
        return Box(low=-3, high=3, shape=(N,), dtype=np.int32)

    @property
    def observation_space(self):
        """See class definition."""
        N = self.initial_vehicles.num_vehicles
        F = 2  # velocity and position

        states = Box(low=-np.inf, high=np.inf, shape=(N, F), dtype=np.float32)
        adjacency = Box(low=0, high=1, shape=(N, N), dtype=np.int32)
        mask = Box(low=0, high=1, shape=(N,), dtype=np.int32)

        return Tuple([states, adjacency, mask])

    def apply_rl_actions(self, rl_actions=None):
        """See class definition."""
        # If it is a tensor data type, it must be converted to a numpy type that sumo can accept
        if isinstance(rl_actions, torch.Tensor):
            rl_actions = rl_actions.detach().cpu().numpy()
        if isinstance(rl_actions, np.ndarray):
            rl_actions2 = rl_actions.copy()

            sorted_rl_ids = [
                veh_id for veh_id in self.sorted_ids
                if veh_id in self.k.vehicle.get_rl_ids()
            ]
            num_hv = self.net_params.additional_params['num_HVs']  # HVs number
            rl_actions2 = rl_actions2[num_hv:num_hv + len(sorted_rl_ids)]  # Reduce the action matrix

            """
                GRL only controls for RL vehicles, so if there are no RL vehicles in the space, 
                the control session is skipped
            """
            if len(sorted_rl_ids) != 0:
                self.k.vehicle.apply_acceleration(sorted_rl_ids, rl_actions2)
            else:
                pass

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if self.env_params.evaluate:
            return np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
        else:
            return rewards.desired_velocity(self, fail=False)

    def get_state(self):
        """See class definition."""
        # speed = [self.k.vehicle.get_speed(veh_id) / self.k.network.max_speed()
        #          for veh_id in self.sorted_ids]
        # pos = [self.k.vehicle.get_x_by_id(veh_id) / self.k.network.length()
        #        for veh_id in self.sorted_ids]

        '''Contructing graph representation'''
        # vehicle numbers
        N = self.initial_vehicles.num_vehicles

        # HVs numbers
        num_HVs = self.net_params.additional_params['num_HVs']

        # CAVs numbers
        num_AVs = self.net_params.additional_params['num_AVs']

        # id acquiring
        ids = self.k.vehicle.get_ids()
        rl_ids = self.k.vehicle.get_rl_ids()
        human_ids = self.k.vehicle.get_human_ids()

        # Initialise the node feature matrix, adjacency matrix and index matrix
        states = np.zeros([N, 2])
        adjacency = np.zeros([N, N])
        mask = np.zeros(N)

        # Constructing graph structure representations when CAVs are in the environment
        if rl_ids:

            # ------Node feature matrix------ #
            # Velocity
            speeds = np.array(self.k.vehicle.get_speed(ids)).reshape(-1, 1)
            speeds_norm = speeds / self.k.network.max_speed()

            # Position
            xs = np.array(self.k.vehicle.get_x_by_id(ids)).reshape(-1, 1)
            xs_norm = xs / self.k.network.length()

            # Feature fusion
            states = np.c_[speeds_norm, xs_norm]

            # ------Adjacency matrix------ #
            # Calculate the horizontal distance between two vehicles
            dist_matrix = euclidean_distances(xs)
            # Generate an all-zero adjacency matrix of the same dimension from dist_matrix
            adjacency_small = np.zeros_like(dist_matrix)
            # Assigning values to the adjacency matrix based on the perceived range of CAVs
            adjacency_small[dist_matrix < 20] = 1
            # Assignment of communication between CAVs in the adjacency matrix
            adjacency_small[-len(rl_ids):, -len(rl_ids):] = 1
            # adjacency = adjacency_small

            # assemble into the NxN adjacency matrix
            adjacency[:len(human_ids), :len(human_ids)] = adjacency_small[:len(human_ids), :len(human_ids)]
            adjacency[num_HVs:num_HVs + len(rl_ids), :len(human_ids)] = adjacency_small[len(human_ids):, :len(human_ids)]
            adjacency[:len(human_ids), num_HVs:num_HVs + len(rl_ids)] = adjacency_small[:len(human_ids), len(human_ids):]
            adjacency[num_HVs:num_HVs + len(rl_ids), num_HVs:num_HVs + len(rl_ids)] = adjacency_small[len(human_ids):,
                                                                                  len(human_ids):]

            # ------Index matrix------ #
            mask[num_HVs:num_HVs + len(rl_ids)] = np.ones(len(rl_ids))

        return states, adjacency, mask

    def additional_command(self):
        """See parent class.

        Define which vehicles are observed for visualization purposes, and
        update the sorting of vehicles using the self.sorted_ids variable.
        """
        # specify observed vehicles
        if self.k.vehicle.num_rl_vehicles > 0:
            for veh_id in self.k.vehicle.get_human_ids():
                self.k.vehicle.set_observed(veh_id)

        # update the "absolute_position" variable
        for veh_id in self.k.vehicle.get_ids():
            this_pos = self.k.vehicle.get_x_by_id(veh_id)

            if this_pos == -1001:
                # in case the vehicle isn't in the network
                self.absolute_position[veh_id] = -1001
            else:
                change = this_pos - self.prev_pos.get(veh_id, this_pos)
                self.absolute_position[veh_id] = \
                    (self.absolute_position.get(veh_id, this_pos) + change) \
                    % self.k.network.length()
                self.prev_pos[veh_id] = this_pos

    @property
    def sorted_ids(self):
        """Sort the vehicle ids of vehicles in the network by position.

        This environment does this by sorting vehicles by their absolute
        position, defined as their initial position plus distance traveled.

        Returns
        -------
        list of str
            a list of all vehicle IDs sorted by position
        """
        if self.env_params.additional_params['sort_vehicles']:
            return sorted(self.k.vehicle.get_ids(), key=self._get_abs_position)
        else:
            return self.k.vehicle.get_ids()

    def _get_abs_position(self, veh_id):
        """Return the absolute position of a vehicle."""
        return self.absolute_position.get(veh_id, -1001)

    def reset(self):
        """See parent class.

        This also includes updating the initial absolute position and previous
        position.
        """
        obs = super().reset()

        for veh_id in self.k.vehicle.get_ids():
            self.absolute_position[veh_id] = self.k.vehicle.get_x_by_id(veh_id)
            self.prev_pos[veh_id] = self.k.vehicle.get_x_by_id(veh_id)

        return obs
