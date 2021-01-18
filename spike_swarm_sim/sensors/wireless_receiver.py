import numpy as np
import numpy.linalg as LA
from spike_swarm_sim.utils import compute_angle
from spike_swarm_sim.register import sensor_registry


def angle_diff(x, y):
    return min((x-y) % (2*np.pi), (y-x) % (2*np.pi))
    
@sensor_registry(name='wireless_receiver_old')
class WirelessReceiver:
    def __init__(self, range=120, n_sectors=8, msg_length=1):
        self.channels = 0
        self.msg = 0
        self.range = range
        self.n_sectors = n_sectors
        self.msg_length = msg_length

    def step(self, my_obj, world_dict, *args):
        pass
        # message_frame = {key : [] for key in ['signal', 'sending_direction', 'receiving_direction', 'msg']}
        # for i, direction in enumerate(self.get_directions(my_obj.theta)):
        #     direction_msgs = []
        #     for obj in world_dict.values():
        #         if my_obj.id != obj.id and type(obj).__name__ == 'Robot':
        #             if 'wireless_transmitter' in obj.actuators:
        #                 v = obj.pos.copy() - my_obj.pos.copy()
        #                 rho = LA.norm(v)
        #                 phi = angle_diff(compute_angle(v), direction)
        #                 cond = np.abs(phi) <= (np.pi / self.n_sectors) + 0.001\
        #                         and rho <= self.range\
        #                         and rho <= obj.actuators['wireless_transmitter'].range\
        #                         and  obj.actuators['wireless_transmitter'].enabled
        #                 if not np.isnan(phi) and cond:
        #                     send_dir = np.argmin([angle_diff(sdir, compute_angle(v) + np.pi) for sdir in self.get_directions(obj.theta)])
        #                     # msg = obj.actuators['wireless_transmitter'].msg[send_dir] #! if directional transmission
        #                     msg = obj.actuators['wireless_transmitter'].msg  #! if isotropic
        #                     send_dir = self.get_directions(0.)[send_dir]
        #                     rx_dir = self.get_directions(0.)[i]
        #                     direction_msgs.append((rho, msg, send_dir, rx_dir))
        #     # Select only the nearest sender from each sector.
        #     active_senders = [(np.exp(-(1/120) * strength), msg, tx_dir, rx_dir)\
        #                     for strength, msg, tx_dir, rx_dir in sorted(direction_msgs, key=lambda v: v[0])][0]\
        #                     if len(direction_msgs) else (0., [0 for _ in range(self.msg_length)], 0., 0.) #! completar con n_bits
            
        #     # if len(active_senders[1]) == 1:
        #     #     active_senders[1] = active_senders[1][0]

        #     message_frame['signal'].append(active_senders[0])
        #     message_frame['msg'].append(active_senders[1])
        #     tx_vec = int(active_senders[0] > 0.) * np.array([np.cos(active_senders[2]), np.sin(active_senders[2])])
        #     message_frame['sending_direction'].append(tx_vec)
        #     rx_vec = int(active_senders[0] > 0.) * np.array([np.cos(active_senders[3]), np.sin(active_senders[3])])
        #     message_frame['receiving_direction'].append(rx_vec)
        #     # import pdb; pdb.set_trace()
        # stochastic_selection = False
        # selected_emitter = np.argmax(message_frame['signal']) #! implementar randomización
        # if any([np.sum(m) > 0 for m in message_frame['msg']]): #* select randomly from frames with non-zero msg
        #     selected_emitter = np.random.choice(np.where([np.sum(m) > 0 for m in message_frame['msg']])[0])
        # elif any(np.array(message_frame['signal']) > 0): #* select randomly from frames with non-zero signal strength
        #     selected_emitter = np.random.choice(np.where(np.array(message_frame['signal']) > 0)[0])
        # #* Add noise to msg
        # # message_frame['msg'][selected_emitter] += 0.2 * np.random.randn(len(active_senders[1]))
        # return {k : np.array(val[selected_emitter]) for k, val in message_frame.items()}
