import numpy as np
from spike_swarm_sim.register import actuator_registry
from spike_swarm_sim.utils import compute_angle

class TargetPositionActuator:
    def __init__(self, ):
        pass

    def step(self, steering):
        self.delta_theta = steering[1]
        self.delta_pos = steering[0]
        print()

