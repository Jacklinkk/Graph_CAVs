import torch

from GRL_Envs.HighwayRamps.HR_base import Env
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from gym.spaces.box import Box
from gym.spaces import Tuple


class MergeEnv(Env):

    @property
    def observation_space(self):
        """Identify the dimensions and bounds of the observation space.
        """
        N = self.net_params.additional_params['num_vehicles']
        F = 2 + self.net_params.additional_params['highway_lanes'] \
            + self.n_unique_intentions

        states = Box(low=-np.inf, high=np.inf, shape=(N, F), dtype=np.float32)
        adjacency = Box(low=0, high=1, shape=(N, N), dtype=np.int32)
        mask = Box(low=0, high=1, shape=(N,), dtype=np.int32)

        return Tuple([states, adjacency, mask])

    @property
    def action_space(self):
        N = self.net_params.additional_params['num_vehicles']
        return Box(low=0, high=1, shape=(N,), dtype=np.int32)
        # return Discrete(3)

    def get_state(self):
        """
            construct a graph representation each time step
        """
        N = self.net_params.additional_params['num_vehicles']
        # num_cav = self.net_params.additional_params['num_cav'] # maximum number of CAVs
        num_hv = self.net_params.additional_params['num_hv']  # maximum number of HDVs

        num_lanes = self.net_params.additional_params['highway_lanes']

        # id acquiring
        vehicle_ids = self.net_params.additional_params['vehicles_ids']

        # CAVs id acquiring
        rl_ids = self.k.vehicle.get_rl_ids()
        # filter the ones on the ramps
        rl_ids = [id_ for id_ in rl_ids if not self.k.vehicle.get_edge(id_).startswith('off_ramp')]
        rl_ids = sorted(rl_ids)
        # print("rl:", rl_ids)

        # HVs id acquiring
        human_ids = sorted(self.k.vehicle.get_human_ids())
        # If too many human ids
        if len(human_ids) > num_hv:
            human_ids = human_ids[:num_hv]

        # Initialise the node feature matrix, adjacency matrix and index matrix
        states = np.zeros([N, 2 + num_lanes + self.n_unique_intentions])
        adjacency = np.zeros([N, N])
        mask_RL = np.zeros(N)

        # when there is rl_vehicles in the scenario
        if rl_ids:
            ids = human_ids + rl_ids

            # Index matrix
            mask_observations = np.zeros(N)
            index_observations = [vehicle_ids.index(elem) for elem in ids]
            mask_observations[index_observations] = 1

            index_RL = [vehicle_ids.index(elem) for elem in rl_ids]
            mask_RL[index_RL] = 1

            # numerical data (speed, location)
            speeds = np.array(self.k.vehicle.get_speed(ids)).reshape(-1, 1)

            # positions = np.array([self.k.vehicle.get_absolute_position(i) for i in ids])  # x y location
            xs = np.array([self.k.vehicle.get_x_by_id(i) for i in ids]).reshape(-1, 1)

            # categorical data 1 hot encoding: (lane location, intention)
            # The number of the lane in which the vehicle is currently located in the environment
            lanes_column = np.array(self.k.vehicle.get_lane(ids))
            # Initialise the lane onehot matrix (current number of vehicles x number of lanes)
            lanes = np.zeros([len(ids), num_lanes])
            # Each vehicle is assigned a value of 1 in the corresponding position of the matrix
            # according to the lane position
            lanes[np.arange(len(ids)), lanes_column] = 1
            # print(np.arange(len(ids)))
            # print("lanes_column", lanes_column)
            # print("lanes:", lanes)

            # intention encoding
            types_column = np.array([self.intention_dict[self.k.vehicle.get_type(i)] for i in ids])
            # Initialise the intention matrix (current number of vehicles x vehicle type)
            intention = np.zeros([len(ids), self.n_unique_intentions])
            # Assign values to the intention matrix according to the type of vehicle in the current environment
            intention[np.arange(len(ids)), types_column] = 1

            # Feature fusion
            observed_states = np.c_[xs, speeds, lanes, intention]

            # assemble into the NxF states matrix
            states[index_observations, :] = observed_states
            states[:, 0] /= self.net_params.additional_params['highway_length']

            # construct the adjacency matrix
            dist_matrix = euclidean_distances(xs)
            # Generate an all-zero adjacency matrix of the same dimension from dist_matrix
            adjacency_small = np.zeros_like(dist_matrix)

            adjacency_small[dist_matrix < 20] = 1
            for count in range(len(index_observations)):  # Connection between CAVs and HVs
                adjacency[index_observations[count], index_observations] = adjacency_small[count, :]
            for count in range(len(index_RL)):  # Connection between CAVs
                adjacency[index_RL[count], index_RL] = 1

            self.observed_cavs = rl_ids
            self.observed_all_vehs = ids

        return states, adjacency, mask_RL

    def compute_reward(self, rl_actions, **kwargs):
        # w_intention = 10
        w_intention = 3
        w_speed = 0.8
        w_p_lane_change = 0.05
        w_p_crash = 0.8
        # w_p_crash = 0

        unit = 1

        # reward for system speed: mean(speed/max_speed) for every vehicle
        speed_reward = 0
        intention_reward = 0

        rl_ids = self.k.vehicle.get_rl_ids()
        if len(rl_ids) != 0:
            # all_speed = np.array(self.k.vehicle.get_speed(self.observed_all_vehs))
            # max_speed = np.array([self.env_params.additional_params['max_hv_speed']]*(len(self.observed_all_vehs) - len(self.observed_cavs))\
            #                     +[self.env_params.additional_params['max_cav_speed']]*len(self.observed_cavs))

            # all_speed = np.array(self.k.vehicle.get_speed(self.observed_cavs))
            all_speed = np.array(self.k.vehicle.get_speed(rl_ids))
            max_speed = self.env_params.additional_params['max_av_speed']
            speed_reward = np.mean(all_speed / max_speed)
            # print("cavs:", self.observed_cavs)
            # print("all_speed:", all_speed)
            # print("speed_reward:", speed_reward)

            ###### reward for satisfying intention ---- only a big instant reward
            # intention_reward = kwargs['num_full_filled'] * unit + kwargs['num_half_filled'] * unit * 0.5
            intention_reward = self.compute_intention_rewards()

        # penalty for frequent lane changing behavors
        # This part calculates the penalty for frequent lane changes
        drastic_lane_change_penalty = 0
        if self.drastic_veh_id:
            drastic_lane_change_penalty += len(self.drastic_veh_id) * unit

        # penalty for crashing
        total_crash_penalty = 0
        crash_ids = kwargs["fail"]
        # print("kwargs: ", kwargs)
        # print("crash:", crash_ids)
        total_crash_penalty = crash_ids * unit
        # print("total_crash_penalty:", total_crash_penalty)
        # if crash_ids:
        #     print(crash_ids,total_crash_penalty)

        # print(speed_reward, intention_reward, total_crash_penalty, drastic_lane_change_penalty)
        return w_speed * speed_reward + \
               w_intention * intention_reward - \
               w_p_lane_change * drastic_lane_change_penalty - \
               w_p_crash * total_crash_penalty

    def compute_intention_rewards(self):  # Intention reward calculation

        intention_reward = 0
        try:
            for cav_id in self.observed_cavs:
                cav_lane = self.k.vehicle.get_lane(cav_id)
                cav_edge = self.k.vehicle.get_edge(cav_id)
                cav_type = self.k.vehicle.get_type(cav_id)
                # print("cav_lane:", cav_lane, "\ncav_edge:", cav_edge, "\ncav_type:", cav_type)

                x = self.k.vehicle.get_x_by_id(cav_id)

                if cav_type == "merge_0":
                    if cav_edge == 'highway_0':
                        val = (self.net_params.additional_params['off_ramps_pos'][0] - x) / \
                              self.net_params.additional_params['off_ramps_pos'][0]
                        if cav_lane == 0:
                            intention_reward += val
                        elif cav_lane == 2:
                            intention_reward -= (1 - val)

                elif cav_type == "merge_1":

                    if cav_edge == "highway_0" and cav_lane == 0:
                        val = (self.net_params.additional_params['off_ramps_pos'][0] - x) / \
                              self.net_params.additional_params['off_ramps_pos'][0]
                        intention_reward += val - 1

                    elif cav_edge == "highway_1":
                        val = (self.net_params.additional_params['off_ramps_pos'][1] - x) / (
                                self.net_params.additional_params['off_ramps_pos'][1] -
                                self.net_params.additional_params['off_ramps_pos'][0])
                        if cav_lane == 0:
                            intention_reward += val
                        elif cav_lane == 2:
                            intention_reward -= (1 - val)

                    else:
                        pass
                else:
                    raise Exception("unknow cav type")
        except:
            pass

        return intention_reward

    def apply_rl_actions(self, rl_actions=None):
        ids = sorted(self.k.vehicle.get_ids())
        rl_ids = self.k.vehicle.get_rl_ids()
        # Remove rl vehicles from rl_ids that have already left the ramp entrance
        rl_ids = [id_ for id_ in rl_ids if not self.k.vehicle.get_edge(id_).startswith('off_ramp')]
        rl_ids = sorted(rl_ids)
        # If it is a tensor data type, it must be converted to a numpy type that sumo can accept
        if isinstance(rl_actions, torch.Tensor):
            rl_actions = rl_actions.detach().cpu().numpy()
        if isinstance(rl_actions, np.ndarray):
            # rl_actions = rl_actions.reshape((self.net_params.additional_params['num_cav'],3))
            rl_actions2 = rl_actions.copy()

            # rl_actions2 -= 1
            rl_actions2 = 1 - rl_actions2

            # rl_ids = self.observed_cavs
            # Calculates the current time and the time interval between the last lane
            # change to detect if the vehicle has made an aggressive change
            drastic_veh = []
            for ind, veh_id in enumerate(rl_ids):
                if rl_actions2[ind] != 0 and (self.time_counter - self.k.vehicle.get_last_lc(veh_id) < 50):
                    drastic_veh.append(veh_id)
                    # print("drastic lane change: ", veh_id)

            self.drastic_veh_id = drastic_veh
            # print("rl_actions2:", rl_actions2)

            # The core controller is in /flow/core/kernel/vehicle/traci.py (apply_lane_change)
            vehicle_ids = self.net_params.additional_params['vehicles_ids']
            index_RL = [vehicle_ids.index(elem) for elem in rl_ids]
            rl_actions2 = rl_actions2[index_RL]
            if len(rl_ids) != 0:
                self.k.vehicle.apply_lane_change(rl_ids, rl_actions2)
            else:
                pass

        return None

    # Counting of RL vehicles successfully exiting the corresponding ramp
    def check_full_fill(self):
        rl_veh_ids = self.k.vehicle.get_rl_ids()
        num_full_filled = 0
        num_half_filled = 0
        for rl_id in rl_veh_ids:
            if rl_id not in self.exited_vehicles:
                current_edge = self.k.vehicle.get_edge(rl_id)
                if current_edge in self.terminal_edges:
                    self.exited_vehicles.append(rl_id)
                    veh_type = self.k.vehicle.get_type(rl_id)

                    # check if satisfy the intention

                    if self.n_unique_intentions == 3:  # specific merge
                        if (veh_type == 'merge_0' and current_edge == 'off_ramp_0') \
                                or (veh_type == 'merge_1' and current_edge == 'off_ramp_1'):
                            num_full_filled += 1
                            print('satisfied: ', rl_id)

                    elif self.n_unique_intentions == 2:  # nearest merge
                        num_full_filled += (current_edge == 'off_ramp_0') * 1
                        num_half_filled += (current_edge == 'off_ramp_1') * 1
                        print("wrongs")

                    else:
                        raise Exception("unknown num of unique n_unique_intentions")
        return num_full_filled, num_half_filled
