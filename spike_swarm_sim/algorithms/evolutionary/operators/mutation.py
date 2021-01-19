
import numpy as np
import pdb
from spike_swarm_sim.register import evo_operator_registry


    
@evo_operator_registry(name='gaussian_mutation')
def gaussian_mutation(population, **kwargs):
    new_pop = []
    for indiv in population:
        mutation_mask = np.random.random(size=indiv.shape) < kwargs['mutation_prob']
        # pdb.set_trace()
        # mutated = indiv.copy()
        # if np.random.random() < mutation_prob:
        #     mutated[np.random.choice(mutated.shape[0])] += np.random.randn() * sigma
        mutated = indiv + mutation_mask * np.random.randn(indiv.shape[0]) * kwargs['sigma']
        new_pop.append(mutated)
    return new_pop

@evo_operator_registry(name='uniform_mutation')
def uniform_mutation(population, **kwargs):
    new_pop = []
    for indiv in population:
        mutation_mask = np.random.random(size=indiv.shape) < kwargs['mutation_prob']
        # pdb.set_trace()
        # mutated = indiv.copy()
        # if np.random.random() < mutation_prob:
        #     mutated[np.random.choice(mutated.shape[0])] += np.random.randn() * sigma
        mutated = indiv + mutation_mask * (np.random.random(size=indiv.shape[0]) * kwargs['max_val']) #! fix for negative values
        new_pop.append(mutated)
    return new_pop

@evo_operator_registry(name='bitflip_mutation')
def bitFlip_mutation(population, **kwargs):
    new_pop = []
    for indiv in population:
        mutation_mask = np.random.random(size=indiv.shape) < kwargs['mutation_prob']
        indiv[mutation_mask] = 1 - indiv[mutation_mask]
        new_pop.append(indiv)
    return new_pop

@evo_operator_registry(name='categorical_mutation')
def categorical_mutation(population, **kwargs):
    new_pop = []
    for indiv in population:
        mutation_mask = np.random.random(size=indiv.shape) < kwargs['mutation_prob']
        indiv[mutation_mask] = np.random.randint(kwargs['min_vals'][mutation_mask], kwargs['max_vals'][mutation_mask])
        new_pop.append(indiv)
    return new_pop