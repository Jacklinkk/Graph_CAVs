# ------Network import------ #
from flow.networks.bottleneck import BottleneckNetwork

name = "bottleneck_example"

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
from flow.networks.bottleneck import ADDITIONAL_NET_PARAMS

print(ADDITIONAL_NET_PARAMS)

# ------Import main parameters------ #
from flow.core.params import NetParams

net_params = NetParams(additional_params=ADDITIONAL_NET_PARAMS)

# ------Import initial parameters------ #
from flow.core.params import InitialConfig

initial_config = InitialConfig(spacing="uniform", perturbation=1)

# ------Import traffic lights parameters------ #
from flow.core.params import TrafficLightParams

traffic_lights = TrafficLightParams()

# ------Import gym environment parameters------ #
from flow.envs.bottleneck import BottleneckEnv

from flow.core.params import SumoParams

sim_params = SumoParams(sim_step=0.1, render=True)

# ------Env construction------ #
from flow.envs.bottleneck import ADDITIONAL_ENV_PARAMS

print(ADDITIONAL_ENV_PARAMS)

from flow.core.params import EnvParams

env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

from flow.core.experiment import Experiment


flow_params = dict(
    exp_tag='bottleneck_example',
    env_name=BottleneckEnv,
    network=BottleneckNetwork,
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

