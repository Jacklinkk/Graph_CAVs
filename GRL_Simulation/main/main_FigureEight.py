from flow.core.params import VehicleParams, SumoCarFollowingParams, SumoParams, EnvParams, \
    InitialConfig, NetParams
from flow.controllers import IDMController, RLController, ContinuousRouter
from GRL_Envs.FigureEight.FE_specific import AccelEnv  # 定义仿真环境
from GRL_Envs.FigureEight.FE_network import FigureEightNetwork, ADDITIONAL_NET_PARAMS  # 定义路网文件

# from controller import SpecificMergeRouter, NearestMergeRouter
# from network import HighwayRampsNetwork, ADDITIONAL_NET_PARAMS

# ----------- Configurations -----------#
# TRAINING = True
TRAINING = False

TESTING = True
# TESTING = False

# Graph configuration
# Enable_Graph = True
Enable_Graph = False

# DEBUG = True
DEBUG = False

RENDER = False
# RENDER = True

# Time horizon
HORIZON = 800

num_AVs = 6
num_HVs = 6

# Boundary of action space
action_min = -3
action_max = 3

# ----------- Simulation -----------#
# 1.Simulation setup
# vehicle numbers
vehicles = VehicleParams()
for i in range(num_HVs):
    vehicles.add(
        veh_id="human{}".format(i),
        acceleration_controller=(IDMController, {"noise": 0.2}),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(speed_mode="obey_safe_speed", decel=1.5),
        num_vehicles=1,
        color="white")
for i in range(num_AVs):
    vehicles.add(
        veh_id="rl{}".format(i),
        acceleration_controller=(RLController, {}),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(speed_mode="obey_safe_speed"),
        num_vehicles=1,
        color="green")

# road network definition
ADDITIONAL_NET_PARAMS["num_AVs"] = num_AVs
ADDITIONAL_NET_PARAMS["num_HVs"] = num_HVs
net_params = NetParams(additional_params=ADDITIONAL_NET_PARAMS)

# Initialization
initial_config = InitialConfig(spacing='uniform')

# 2.Simulation environment
# parameter setting
env_params = EnvParams(horizon=HORIZON,
                       additional_params={
                           "target_velocity": 40,
                           "max_accel": action_max,
                           "max_decel": -action_min,
                           "sort_vehicles": False})

# SUMO setting
sim_params = SumoParams(sim_step=0.1, restart_instance=True, render=RENDER)

# ----------- Model Building -----------#
flow_params = dict(
    exp_tag='figure_eight',
    env_name=AccelEnv,
    network=FigureEightNetwork,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config
)

# simulation start
from GRL_Experiment.Exp_FigureEight.FE_TD3 import Experiment

exp = Experiment(flow_params)
# run the sumo simulation
exp.run(num_HVs=num_HVs, num_AVs=num_AVs,
        training=TRAINING, testing=TESTING,
        action_min=action_min, action_max=action_max,
        Graph=Enable_Graph)
