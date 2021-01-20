import pdb
import random 
import numpy as np
from spike_swarm_sim.register import evo_operator_registry

@evo_operator_registry(name='uniform_crossover')
def uniform_crossover(parents, random_pairs=False, crossover_prob=1.):
    offspring = []
    if random_pairs: np.random.shuffle(parents)
    if len(parents) % 2: offspring.append(parents.pop(0))
    for parent1, parent2  in zip(parents[::2], parents[1::2]):
        crossover_mask = np.random.randint(2, size=parent1.shape[0])
        new1 = parent1 * crossover_mask + parent2 * (1 - crossover_mask)
        new2 = parent2 * crossover_mask + parent1 * (1 - crossover_mask)
        do_crossover = np.random.random() < crossover_prob
        offspring.append((parent1, np.hstack(new1))[do_crossover])
        offspring.append((parent2, np.hstack(new2))[do_crossover])
    return offspring

@evo_operator_registry(name='onepoint_crossover')
def onepoint_crossover(parents, random_pairs=False, crossover_prob=1.):
    offspring = []
    if random_pairs: np.random.shuffle(parents)
    if len(parents) % 2: offspring.append(parents.pop(0))
    for parent1, parent2 in zip(parents[::2], parents[1::2]):
        cut_idx = np.random.randint(parent1.shape[0])
        new1 = np.hstack((parent1[:cut_idx], parent2[cut_idx:]))
        new2 = np.hstack((parent2[:cut_idx], parent1[cut_idx:]))
        do_crossover = np.random.random() < crossover_prob
        offspring.append((parent1, np.hstack(new1))[do_crossover])
        offspring.append((parent2, np.hstack(new2))[do_crossover])
    return offspring

@evo_operator_registry(name='blxalpha_crossover')
def blxalpha_crossover(parents, random_pairs=False, crossover_prob=1., alpha=.3):
    offspring = []
    if random_pairs: np.random.shuffle(parents)
    if len(parents) % 2: offspring.append(parents.pop(0))
    for parent1, parent2 in zip(parents[::2], parents[1::2]):
        genes_min = np.min((parent1, parent2), axis=0) - alpha * np.abs(parent1-parent2)
        genes_max = np.max((parent1, parent2), axis=0) + alpha * np.abs(parent1-parent2)
        new1 = np.random.random(size=parent1.shape) * (genes_max - genes_min) + genes_min
        new2 = np.random.random(size=parent2.shape) * (genes_max - genes_min) + genes_min
        do_crossover = np.random.random() < crossover_prob
        offspring.append((parent1, np.hstack(new1))[do_crossover])
        offspring.append((parent2, np.hstack(new2))[do_crossover])
    return offspring


@evo_operator_registry(name='combination_crossover')
def combination_crossover(parents, random_pairs=False, crossover_prob=1., alpha=.3):
    offspring = []
    if random_pairs: np.random.shuffle(parents)
    if len(parents) % 2: offspring.append(parents.pop(0))
    for parent1, parent2 in zip(parents[::2], parents[1::2]):
        genes_min = np.min((parent1, parent2), axis=0) - alpha * np.abs(parent1-parent2)
        genes_max = np.max((parent1, parent2), axis=0) + alpha * np.abs(parent1-parent2)
        new1 = np.random.random(size=parent1.shape) * (genes_max - genes_min) + genes_min
        new2 = np.random.random(size=parent2.shape) * (genes_max - genes_min) + genes_min
        do_crossover = np.random.random() < crossover_prob
        offspring.append((parent1, np.hstack(new1))[do_crossover])
        offspring.append((parent2, np.hstack(new2))[do_crossover])
    return offspring

@evo_operator_registry(name='multipoint_crossover')
def multipoint_crossover(parents, ncuts=3, crossover_prob=1.):
    offspring = []
    if len(parents) % 2: 
        offspring.append(parents.pop(np.random.choice(len(parents))))
    for parent1, parent2  in zip(parents[::2], parents[1::2]):
        cut_idxs = np.sort(np.random.choice(range(1, parent1.shape[0]-1), size=ncuts, replace=False))
        cut_idxs = np.hstack(([0], cut_idxs, [None]))
        new1 = []; new2 = []
        for ii, cut_idx in enumerate(cut_idxs[:-1]):
            if ii % 2 == 0:
                new1.append(parent1[cut_idx:cut_idxs[ii+1]])
                new2.append(parent2[cut_idx:cut_idxs[ii+1]])
            else:
                new1.append(parent2[cut_idx:cut_idxs[ii+1]])
                new2.append(parent1[cut_idx:cut_idxs[ii+1]])
        do_crossover = np.random.random() < crossover_prob
        offspring.append((parent1, np.hstack(new1))[do_crossover])
        offspring.append((parent2, np.hstack(new2))[do_crossover])
    return offspring


@evo_operator_registry(name='simulated_binary_crossover')
def simulated_binary_crossover(parents, eta=.5, crossover_prob=1.):#! Mirar valores eta
    offspring = []
    if len(parents) % 2:
        offspring.append(parents.pop(np.random.choice(len(offspring))))
    for parent1, parent2 in zip(parents[::2], parents[1::2]):
        mu = np.random.random()
        beta = ((2*mu, .5/(1-mu))[mu >= .5]) ** (1/(eta+1))
        new1 = .5 * ((1+beta) * parent1 + (1-beta) * parent2)
        new2 = .5 * ((1-beta) * parent1 + (1+beta) * parent2)
        do_crossover = np.random.random() < crossover_prob
        offspring.append((parent1, np.hstack(new1))[do_crossover])
        offspring.append((parent2, np.hstack(new2))[do_crossover])
    return offspring

@evo_operator_registry(name='pcx_crossover')
def pcx_crossover(parents, crossover_prob=1.):
    #TODO
    raise NotImplementedError 


def combined_crossover(parents, eta=1., crossover_prob=1.):
    #TODO
    raise NotImplementedError 


# def blxalpha_beta_crossover(parents, random_pairs=False, crossover_prob=1., alpha=.3):
#     offspring = []
#     if random_pairs: np.random.shuffle(parents)
#     if len(parents) % 2: offspring.append(parents.pop(0))
#     for parent1, parent2 in zip(parents[::2], parents[1::2]):
#         genes_min = np.min((parent1, parent2),axis=0)-alpha*np.abs(parent1-parent2)
#         genes_max = np.max((parent1, parent2),axis=0)+alpha*np.abs(parent1-parent2)
#         new1 = np.random.random(size=parent1.shape)*(genes_max - genes_min) + genes_min
#         new2 = np.random.random(size=parent2.shape)*(genes_max - genes_min) + genes_min
#         do_crossover = np.random.random() < crossover_prob
#         offspring.append((parent1, np.hstack(new1))[do_crossover])
#         offspring.append((parent2, np.hstack(new2))[do_crossover])
#     return offspring