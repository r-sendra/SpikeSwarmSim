import numpy as np
import numpy.linalg as LA
from spike_swarm_sim.utils import compute_angle, normalize
from spike_swarm_sim.register import sensor_registry

@sensor_registry(name='neighbors_state_sensor')
class NeighborsStateSensor:
    def __init__(self,):
        self.range = 200
        self.msg_length = 8
        
    def step(self, my_obj, world_dict, *args):
        state_aggr = []
        distances = []
        for obj in world_dict.values():
            if my_obj.id != obj.id and type(obj).__name__ == 'Robot':
                if 'wireless_transmitter' in obj.actuators:
                    v = obj.pos.copy() - my_obj.pos.copy()
                    R = LA.norm(v)
                    cond = R <= self.range and R <= obj.actuators['wireless_transmitter'].range
                    if cond:
                        if obj.actuators['wireless_transmitter'].msg is not None:
                            state_aggr.append(obj.actuators['wireless_transmitter'].msg)
                        else:
                            state_aggr.append(np.zeros(self.msg_length))
                        distances.append(R)
        if len(distances):
            distances = np.stack(distances)/ sum(distances)
            # print(np.mean(state_aggr))
            # import pdb; pdb.set_trace()
            state_aggr = distances.dot(np.stack(state_aggr)) + np.random.randn(len(state_aggr[0]))*.1
        else:
            state_aggr = np.zeros(self.msg_length)
        return state_aggr