import numpy as np
import torch 
import time
import random
import pdb
import multiprocessing
import copy
import signal
import pickle
import matplotlib.pyplot as plt
from .operators import *
class KeyboardInterruptError(Exception): pass

LOAD = 0
class CoEvoAlgorithm:
    def __init__(self, world,   
                n_generations=100, 
                population_size=100, 
                mutation_prob=0.05, 
                elitism_ratio=0.05,
                eval_steps=500,
                crossover_operator='blxalpha',
                mutation_operator='gaussian',
                selection_squeme='roulette',
                n_processes=1,
                fitness_fn=None):
        
        self.n_generations = n_generations
        self.population_size = population_size
        self.mutation_prob = mutation_prob
        self.elitism_ratio = elitism_ratio
        self.eval_steps = eval_steps
        self.n_processes = n_processes

        self.crossover_operator = {
            'uniform' : uniform_crossover,
            'onepoint' : onepoint_crossover,
            'multipoint' : multipoint_crossover,
            'blxalpha' : blxalpha_crossover,
        }[crossover_operator]
        
        self.mutation_operator = {
            'gaussian' : gaussian_mutation,
        }[mutation_operator]

        self.selection_squeme = {
            'roulette' : roulette_selection,
            'tournament' : tournament_selection,
        }[selection_squeme]

        self.fitness_fn = fitness_fn
        if fitness_fn is None: raise Exception('Error: Specify fitness function.')
        self.robots = [copy.deepcopy(robot) for _, robot in world.flattened_dict.values() if robot.trainable]
        self.worlds = [copy.deepcopy(world) for _ in range(self.n_processes)]
        self.genome_shape = self.robots[0].controller.neural_network.toGenotype()['genotype'].shape[0]
        self.population = {robot._id : [] for robot in self.robots} # dict storing all subpops 
        self.population_info = {}
        self.fitness =  [0 for _ in range(self.population_size)] 
        self.evolution_history = {stat : [] for stat in ['mean', 'max', 'min']}
        self.init_generation = 0
        
        if LOAD:
            self.load_population()
        else:
            self.initialize_population()

    def run(self):
        for k in range(self.init_generation, self.n_generations):
            fitness = []
            t0 = time.time()
            if self.n_processes > 1:
                original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
                signal.signal(signal.SIGINT, original_sigint_handler)
                pool = multiprocessing.Pool(processes=self.n_processes)
                for i_batch in range((self.population_size) // self.n_processes):
                    try:
                        fitness_batch = pool.map_async(self._run_swarm,\
                                    np.arange(i_batch*self.n_processes, (i_batch+1)*self.n_processes)).get(10e5)
                        fitness_batch = [v for _, v in sorted(fitness_batch, key=lambda x: x[0])]
                        fitness.extend(fitness_batch)
                    except KeyboardInterrupt:
                        print('got ^C while pool mapping, terminating the pool')
                        pool.terminate(); exit(1)
                    except Exception as e:
                        print('got exception: ',e,', terminating the pool')
                        pool.terminate(); exit(1)
                pool.terminate()
                self.fitness = fitness
            else:
                fitness = [self._run_swarm(i) for i in range(self.population_size)]
                self.fitness = [v for _, v in fitness]  
            mean_fitness, max_fitness, min_fitness = self.evolve()
            self.mutation_prob = self.mutation_prob_schedule(k)
            print('End of generation ', k, ' with mean fitness: ', mean_fitness, ' and max fitness ', \
                        max_fitness, 'Mutation ratio ', self.mutation_prob, 'elapsed time: ', time.time()-t0)
            any([self.evolution_history[stat_name].append(stat) for stat_name, stat in \
                        zip(['mean', 'max', 'min'], [mean_fitness, max_fitness, min_fitness])])
            if k % 5 == 0:
                self.save_population(k)
            
    def _run_swarm(self, env_id):
        try:
            world = copy.deepcopy(self.worlds[env_id % self.n_processes])
            world.reset()
            robots = [robot for _, robot in world.flattened_dict.values() if robot.trainable]
            beacons = [robot for _, robot in world.flattened_dict.values() if robot.controllable and not robot.trainable]
            genomes = [self.population[robot._id][env_id] for robot in robots]
            any([robot.controller.neural_network.fromGenotype(genome) \
                    for robot, genome in zip(robots, genomes)])

            fitness = np.zeros(len(robots))
            eval_hist = {'actions': [], 'states': []}
            for eval_step in range(self.eval_steps):
                states, actions = world.step()
                eval_hist['actions'].append(actions)
                eval_hist['states'].append(states)
            fitness = self.fitness_fn(eval_hist['actions'], eval_hist['states'])
            return  (env_id, fitness)
        except KeyboardInterrupt:
            raise KeyboardInterruptError()

    def evolve(self):
        parents = {robot._id : self.selection_squeme(self.population[robot._id].copy(), \
                        self.fitness.copy(), np.ceil(self.population_size*(1-self.elitism_ratio)).astype(int))\
                        for robot in self.robots}
        offspring = {robot._id : self.crossover_operator(parents[robot._id], crossover_prob=.9)\
                    for robot in self.robots}
        offspring = {robot._id : self.mutation_operator(offspring[robot._id],\
                    mutation_prob=self.mutation_prob, sigma=.1,clip_range=\
                    (self.population_info['numerical_genes']['constrains']['min'],\
                    self.population_info['numerical_genes']['constrains']['max']))
                    for robot in self.robots}

        fitness_orders =  np.argsort(self.fitness)[::-1] 
        elite = {robot._id : [self.population[robot._id][idx].copy()\
                for idx in fitness_orders[:int(self.population_size\
                 * self.elitism_ratio)]] for robot in self.robots}
        self.population = {robot._id : elite[robot._id] + offspring[robot._id] \
                            for robot in self.robots}
        mean_fitness = np.mean(self.fitness)
        max_fitness = np.max(self.fitness)
        min_fitness = np.min(self.fitness)

        self.fitness = [0 for _ in range(self.population_size)]
        return mean_fitness, max_fitness, min_fitness 

    def save_population(self, generation):
        pop_checkpoint = {
            'population' : self.population,
            'population_info' : self.population_info,
            'generation' : generation,
            'mutation_prob' : self.mutation_prob,
            'evolution_hist' : self.evolution_history,
        }
        torch.save(pop_checkpoint, 'spike_swarm_sim/checkpoints/populations/checkpoint.pth.tar')

    def load_population(self):
        checkpoint = torch.load('spike_swarm_sim/checkpoints/populations/checkpoint.pth.tar')#!
        self.population = {robot._id : subpop for robot, subpop in zip(self.robots, checkpoint['population'].values())}
        self.population_size = len(tuple(self.population.values())[0])
        self.mutation_prob = checkpoint['mutation_prob']
        self.population_info = checkpoint['population_info']
        self.init_generation = checkpoint['generation']
        self.evolution_history = checkpoint['evolution_hist']

    def initialize_population(self):
        for robot in self.robots:
            for i in range(self.population_size):
                robot.controller.neural_network.initialize()
                self.population[robot._id].append(robot.controller.neural_network.toGenotype()['genotype'])
        genotype_dict = self.robots[0].controller.neural_network.toGenotype()
        self.population_info.update({
            'categorical_genes': genotype_dict['categorical_genes'],
            'numerical_genes' :  genotype_dict['numerical_genes'],
        })

    def evaluate(self):
        best = self.population[0]
        world = self.worlds[0]
        robots = [robot for _, robot in world.flattened_dict.values() if robot.trainable]
        # genomes = self.population[int(env_id*len(self.robots)):int((env_id+1)*len(robots))]
        world.reset()
        for robot in robots:
            robot.controller.neural_network.fromGenotype(best[robot._id])
        fitness = np.zeros(len(robots))
        eval_hist = {'actions': [], 'states': []}
        A = []
        for eval_step in range(self.eval_steps):
            states, actions = world.step()
            A.append(actions)
            eval_hist['actions'].append(actions)
            eval_hist['states'].append(states)
            
        fitness = self.fitness_fn(eval_hist['actions'], eval_hist['states'])
        return [fitness for _ in range(len(robots))]


    def plot_learning_curve(self):
        plt.plot(self.evolution_history['mean'])
        plt.fill_between(range(len(self.evolution_history['mean'])),\
            self.evolution_history['min'], self.evolution_history['max'], color='blue', alpha=.1)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.show()

    def mutation_prob_schedule(self, k): 
        return (0.7 - 0.05) * np.exp(-k/10) + 0.05
