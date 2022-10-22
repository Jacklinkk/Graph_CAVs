# ------Network import------ #
from flow.networks.highway import HighwayNetwork

name = "highway_example"

# ------Vehicle parameters------ #
# 1.Import vehicle parameters
from flow.core.params import VehicleParams

vehicles = VehicleParams()

# 2.Import vehicle model
from flow.controllers.car_following_models import IDMController

from flow.controllers.routing_controllers import ContinuousRouter

vehicles.add("human",
             acceleration_controller=(IDMController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=10)


# ------Import network parameters------ #
from flow.networks.highway import ADDITIONAL_NET_PARAMS

# print(ADDITIONAL_NET_PARAMS)

ADDITIONAL_NET_PARAMS2 = {
    # length of the highway
    "length": 300,
    # number of lanes
    "lanes": 3,
    # speed limit for all edges
    "speed_limit": 30,
    # number of edges to divide the highway into
    "num_edges": 1,
    # whether to include a ghost edge. This edge is provided a different speed
    # limit.
    "use_ghost_edge": False,
    # speed limit for the ghost edge
    "ghost_speed_limit": 25,
    # length of the downstream ghost edge with the reduced speed limit
    "boundary_cell_length": 300
}
print(ADDITIONAL_NET_PARAMS2)

# ------Import main parameters------ #
from flow.core.params import NetParams

net_params = NetParams(additional_params=ADDITIONAL_NET_PARAMS2)

# ------Import initial parameters------ #
from flow.core.params import InitialConfig

initial_config = InitialConfig(spacing="uniform")

# ------Import traffic lights parameters------ #
from flow.core.params import TrafficLightParams

traffic_lights = TrafficLightParams()

# ------Import gym environment parameters------ #
from flow.envs.ring.accel import AccelEnv

from flow.core.params import SumoParams

sim_params = SumoParams(sim_step=0.1, render=True)

from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS

print(ADDITIONAL_ENV_PARAMS)

from flow.core.params import EnvParams

env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

from flow.core.experiment import Experiment

# network = RingNetwork("ring", vehicles, net_params)

flow_params = dict(
    exp_tag='highway_example',
    env_name=AccelEnv,
    network=HighwayNetwork,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
    tls=traffic_lights,
)

# number of time steps
flow_params['env'].horizon = 3000
exp = Experiment(flow_params)

# run the sumo simulation
_ = exp.run(1, convert_to_csv=False)

