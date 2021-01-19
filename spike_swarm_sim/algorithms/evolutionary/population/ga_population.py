from .population import Population

class GA_Population(Population):
    def __init__(self, *args, **kwargs):
        super(GA_Population, self).__init__(\
            neural_network=CTRNNet(topology, debug_options=debug_options), \
            *args, **kwargs)
    
    def step(self, fitness_vector):
        # Apply Selection operator
        parents, parents_fitness = self.selection_operator(self.population.copy(), fitness_vector.copy(),\
                    len(self) - self.num_elite)
        # Apply Mating operator
        parents = self.mating_operator(parents, parents_fitness)
         
        # Apply Crossover operator
        offspring = self.crossover_operator(parents, crossover_prob=self.crossover_prob)
        
        # Apply mutation operator
        offspring = self.mutation_operator(offspring, mutation_prob=self.mutation_prob, sigma=.1,\
                             max_vals=self.max_vector, min_vals=self.min_vector)

        # Constrain values
        if self.encoding == 'real':
            # pdb.set_trace()
            offspring = [np.clip(v, a_min=self.min_vector, a_max=self.max_vector) for v in offspring] 

        # Save elite based on highest fitness
        fitness_order = np.argsort(fitness_vector.copy())[::-1]     
        elite = [self.population[idx].copy() for idx in fitness_order[:self.num_elite]]

        # Update new population
        self.population = elite + offspring

        # Dynamic Mutation Prob.
        # self.mutation_prob = exp_schedule(self.mutation_prob, 0.01)