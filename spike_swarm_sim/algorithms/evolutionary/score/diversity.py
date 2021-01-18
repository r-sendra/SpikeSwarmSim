from collections import deque, Counter
import numpy as np
from spike_swarm_sim.utils import key_of, compute_angle


def euclidean_distance(v1, v2):
    return np.linalg.norm(v1-v2) / (1.5*np.sqrt(v1.shape[0]))  

class DiversityScore:
    def __init__(self, dist_fn='cosine', alpha=1, rad=2):
        self.alpha = alpha
        self.rad = rad
    def __call__(self, indiv_id, population):
        indiv = population[indiv_id]
        #! FITNESS SHARING
        # return self.importance * (1-np.exp(-.01*(np.sum([np.linalg.norm(indiv - v)\
            #  for v in population])/(len(population)-1))**2))
        f_sharing = np.mean([(0, 1-(euclidean_distance(indiv, v)/self.rad)**self.alpha)\
                    [np.sqrt(euclidean_distance(indiv, v)) < self.rad]\
                    for i, v in enumerate(population) if i != indiv_id])
        return 1 - f_sharing

clusters = np.array([[xa, xb] for xa, xb in \
            zip(*map(lambda v: v.flatten(),
            np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4))))])



class BehaviouralDiversity:
    def __init__(self):
        self.ids = []
        self.beh_div = []
        self.dist_walked = deque([])
        self.times_symbols = deque([])
        self.times_res = deque([])
        self.mean_vel = deque([])
        self.mean_neighs = deque([])
        self.mean_light = deque([])

    def __call__(self, actions, states, info=None):
        # positions = np.stack(info["robot_positions"]).copy()
        # light_positions = np.stack(info["light_positions"]).copy()
        self.ids.append(id)
        
        # #* Distance walked
        # dist_walked = np.mean([np.abs(start - end) / 500 for start, end in zip(positions[0], positions[-1])], 0)

        #* Times each symbol
        # sym_dict = {i : sym for i, sym in enumerate(clusters)}
        # times_symbols = np.zeros([16, 16]) #{k : 0 for k in sym_dict.keys()}
        # for t in range(1, len(actions)):
        #     action_t = actions[t]
        #     state_t = states[t]
        #     for ac_robot, st_robot in zip(action_t, state_t):
        #         symbol_out = key_of(sym_dict, ac_robot['wireless_transmitter']['msg'])
        #         symbol_in = key_of(sym_dict, st_robot['wireless_receiver']['msg'])
        #         times_symbols[symbol_in, symbol_out] += 1
        # # times_symbols /= ((len(states) - 1) * len(states[0]))
        # # num_samples = np.sum(times_symbols, 0)
        # times_symbols /= times_symbols.max() #num_samples[num_samples > 0]
        # times_symbols = times_symbols.T.flatten()

        #* Symbol followed (assume 16 symbols)
        
        # symb_followed = {k : np.zeros(16) for k in sym_dict.keys()}
        # for t in range(5, len(actions)):
        #     state_prev = states[t-5]
        #     ac_t = actions[t]
        #     for ac_robot, st_robot in zip(ac_t, state_prev):
        #         ac_index = key_of(sym_dict, ac_robot['wireless_transmitter']['msg'])
        #         st_index = key_of(sym_dict, st_robot['wireless_receiver']['msg'])
        #         symb_followed[st_index][ac_index] += 1
        # symb_followed = np.array([np.argmax(syms)/15 for syms in symb_followed.values()]) 
        

        # #* Times response
        # times_res=np.mean([ac_robot['wireless_transmitter']['priority'] for action_t in actions for ac_robot in action_t], 0)

        #* Average velocity
        # mean_vel = np.mean([ac_robot['wheel_actuator'] for action_t in actions for ac_robot in action_t], 0)

        # #* Average number of neighbours
        # neighs=np.mean([np.mean([np.linalg.norm(posi - posj) < 100 for posi in positions[-1]]) for posj in positions[-1]])

        #* Average distance to neighbours and light
        # #! USED for light exps
        # angles_light = np.zeros(len(positions[0]))
        # for t, (pos, light_pos) in enumerate(zip(positions, light_positions)):
        #     angles_light += np.array([compute_angle((pi - light_pos).flatten()) / (2*np.pi) for pi in pos])
        # angles_light /= len(positions)

        #* Num robots in light at the end
        # times_light = np.zeros(len(positions[0]))
        # for t, (pos, light_pos) in enumerate(zip(positions, light_positions)):
        #     times_light += np.array([float(np.linalg.norm(pi - light_pos) <= 100) for pi in pos])
        # times_light /= len(positions)

        #* COMM MEANS 
        msg_len = len(actions[0][0]['wireless_transmitter']['msg'])
        symbols = [centroid for centroid in zip(*map(lambda v: v.flatten(),\
            np.meshgrid(*[np.linspace(0, 1, 4) for _ in np.arange(msg_len)])))]
        # symbols = np.array(symbols)
        symbols = {tuple(sym) : i for i, sym in enumerate(symbols)}
        counter_relay = Counter({(sym_in, led) : 0  for led in [0, 1] for sym_in in symbols.values()})#np.zeros(16*2+2)
        counter_bcast = Counter({(sym_in, sym_out, led) : 0  for led in [0, 1] for sym_in in symbols.values() for sym_out in symbols.values()})#np.zeros(16*2+2)
        counter_ledON = Counter({(sym_in, sym_out, st) : 0  for st in [0, 1] for sym_in in symbols.values() for sym_out in symbols.values()})
        counter_ledOFF = Counter({(sym_in, sym_out, st) : 0  for st in [0, 1] for sym_in in symbols.values() for sym_out in symbols.values()})
        counter_symbols = {i : Counter({(sym_in, led, st) : 0  for led in [0, 1] for st in [0, 1] for sym_in in symbols.values()}) for i in range(len(symbols))}
        # most_common_ledOFF = np.zeros(3)
        for t, (actions_t, states_t) in enumerate(zip(actions, states)):
            commst_v = np.stack([ac['wireless_transmitter']['state'] for ac in actions_t])
            Sout_v = np.stack([symbols[tuple(ac['wireless_transmitter']['msg'])] for ac in actions_t])
            Sin_v = np.stack([symbols[tuple(st['wireless_receiver']['msg'])] for st in states_t])
            led_v = np.stack([ac['led_actuator'] for ac in actions_t])
            # --- ADD MOST FREQUENT SIN, SOUT and LED for STATE=RELAY
            if any(commst_v == 0):
                for sin, led in zip(Sin_v[commst_v == 0], led_v[commst_v == 0]):
                    counter_relay.update([(sin, led)])
            if any(commst_v == 1):
                for sin, sout, led in zip(Sin_v[commst_v == 1], Sout_v[commst_v == 1], led_v[commst_v == 1]):
                    counter_bcast.update([(sin, sout, led)])
            if any(led_v == 0):
                for sin, sout, st in zip(Sin_v[led_v == 0], Sout_v[led_v == 0], commst_v[led_v == 0]):
                    counter_ledOFF.update([(sin, sout, st)])
            if any(led_v == 1):
                for sin, sout, st in zip(Sin_v[led_v == 1], Sout_v[led_v == 1], commst_v[led_v == 1]):
                    counter_ledON.update([(sin, sout, st)])
            for sym in symbols.values():
                if any(Sout_v == sym):
                    for sin, led, st in zip(Sin_v[Sout_v == sym], led_v[Sout_v == sym], commst_v[Sout_v == sym]):
                        counter_symbols[sym].update([(sin, led, st)])
            # import pdb; pdb.set_trace()
            # ac_comm_st = np.stack([ac['wireless_transmitter']['state'] for ac in actions_t])
            # times_relay += 1 - ac_comm_st
        comm_behavior = np.hstack([cnt.most_common(1)[0][0] if cnt.most_common(1)[0][1] > 0 else tuple(-1 for _ in cnt.most_common(1)[0][0]) \
                     for cnt in [counter_relay, counter_bcast, counter_ledOFF, counter_ledON]+[*counter_symbols.values()]])
        # print(time.time() - t0)
        # import pdb; pdb.set_trace()
        # most_common_relay = couter_relay.most_common(1)[0][0]
        # times_relay /= len(actions)

        # #* TIMES LEADERS
        # n_robots = len(actions[0])
        # n_steps = len(actions)
        # times_leaders = np.zeros(len(actions[0])) # for each robot, the maximum consec. time steps being leader.
        # consec_times_leader = 0
        # prev_leader = None
        # for t, (actions_t) in enumerate(actions):
        #     ac_leds = np.stack([ac['led_actuator'] for ac in actions_t])
        #     if ac_leds.sum() == 1: # leader ok
        #         new_leader = np.argmax(ac_leds)
        #         if prev_leader == new_leader:
        #             consec_times_leader += 1
        #             if times_leaders[new_leader] < consec_times_leader: # if it is a record
        #                 times_leaders[new_leader] = consec_times_leader # update score
        #         else:
        #             consec_times_leader = 0
        #         prev_leader = new_leader
        #     else:
        #         prev_leader = None
        #         consec_times_leader = 0
        # times_leaders = np.sort(times_leaders) / 80 # 80 is the max timesteps a robot can be leader.
        # # times_leaders = np.mean(times_leaders) / ((80 * (n_steps//80)) / n_robots) if any(times_leaders > 0) else 0 # it is a scalar
        # import pdb; pdb.set_trace()
        # return times_leaders #  np.hstack([var for var in [angles_light]])#!
        return comm_behavior


    # def __call__(self, indiv_id, pop_size):
    #     archive_idx = np.where(np.array(self.ids) == indiv_id)[0][0]
    #     behavioral_div = 0
    #     for ind in len(pop_size):
    #         if ind == indiv_id:
    #             continue
    #         vecA1 = np.hstack([var[archive_idx] for var in [self.dist_walked, self.times_res, self.mean_vel, self.mean_neighs, self.mean_light]])
    #         vecB1 = self.times_symbols[archive_idx]

    #         vecA2 = np.hstack([var[ind] for var in [self.dist_walked, self.times_res, self.mean_vel, self.mean_neighs, self.mean_light]])
    #         vecB2 = self.times_symbols[ind]
    #         import pdb; pdb.set_trace()
    #         behavioral_div += 0.5 * (np.linalg.norm(vecA1 - vecA2) + np.linalg.norm(vecB1 - vecB2))
    #     return behavioral_div / (pop_size - 1)