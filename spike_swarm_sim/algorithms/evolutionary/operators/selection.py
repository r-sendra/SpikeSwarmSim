import pdb
import numpy as np
from spike_swarm_sim.register import evo_operator_registry

@evo_operator_registry(name='roulette_selection')
def roulette_selection(population, fitness, n_sel):
    probs = [f / np.sum(fitness) for f in fitness]
    sel_idxs = np.random.choice(len(population), p=probs, replace=True, size=n_sel)
    selected = [population[i] for i in sel_idxs]
    fitness_sel = [fitness[i] for i in sel_idxs]
    # selected = [v for v, _ in sorted(zip(selected, fitness_sel), key=lambda x: x[1])]
    return selected, fitness_sel

@evo_operator_registry(name='lin_rank_selection')
def lin_rank_selection(population, fitness, n_sel, sp=1.5):
    fitness_order = np.argsort(fitness)
    population = [population[ii] for ii in fitness_order]
    # * First individual is the worst
    probs = [2-sp+2*(sp-1)*i/(len(population)-1) for i in range(len(population))]
    probs[0] += 1-sum(probs)

    sel_idxs = np.random.choice(len(population), replace=True, size=n_sel)
    selected = [population[i] for i in sel_idxs]
    fitness_sel = [fitness[i] for i in sel_idxs]
    # selected = [v for v, _ in sorted(zip(selected, fitness_sel), key=lambda x: x[1])]
    return selected, fitness_sel

@evo_operator_registry(name='nonlin_rank_selection')
def nonlin_rank_selection(population, fitness, n_sel, p_best=.1):
    fitness_order = np.argsort(fitness)[::-1]
    population = [population[ii] for ii in fitness_order]
    probs = [p_best*(1-p_best)**i for i in range(len(population))]
    probs[0] += 1-sum(probs)
    sel_idxs = np.random.choice(len(population), p=probs, replace=True, size=n_sel)
    selected = [population[i] for i in sel_idxs]
    fitness_sel = [fitness[i] for i in sel_idxs]
    # selected = [v for v, _ in sorted(zip(selected, fitness_sel), key=lambda x: x[1])]
    return selected, fitness_sel


@evo_operator_registry(name='tournament_selection')
def tournament_selection(population, fitness, n_sel, tournament_size=3):
    selected = []
    selected_fitness = []
    for _ in range(n_sel):
        competitors_idx = np.random.choice(len(population), replace=True, size=tournament_size)
        competitors = [population[idx] for idx in competitors_idx]
        competitors_fitness = [fitness[idx] for idx in competitors_idx]
        selected.append(competitors[np.argmax(competitors_fitness)])
        selected_fitness.append(np.max(competitors_fitness))
    # selected = [v for v, _ in sorted(zip(selected, selected_fitness), key=lambda x: x[1])]
    return selected, selected_fitness


@evo_operator_registry(name='stochastic_universal_selection')
def stochastic_universal_selection(population, fitness, n_sel):
    P = sum(fitness) / n_sel
    start = np.random.random()*P
    pointers = [start + i*P for i in range(n_sel)] 
    selected = []
    selected_fitness = []
    for pt in pointers:
        i = 0
        while sum(fitness[:i] < pt): i+=1 
        selected.append(population[i])
        selected_fitness.append(fitness[i])
    return selected,selected_fitness