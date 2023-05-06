import random
import time
from Portfolio_Simulation import Simulator

class GAOptimizer():

	def __init__(self, strat, fitness_criterion, simulator, population_size, num_generations, k, p_cross, p_mut, **kwargs):

		'''
		Inputs:
		- strat: list of strategy components (parameters & potential values, signal function)
		- fitness_criterion: choice of measurement of fitness (min volatility, max sharpe, max sortino, max calmar)
		- simulator: simulator to use during training and optimization
		- population_size: size of each generation
		- num_generations: number of times to run the optimizer
		- k: sample size for tournament selection
		- p_cross: probability of crossover
		- p_mut: probability of mutation
		- **kwargs: sector constraints for mean-variance portfolio construction (if any)
		'''
		
		self.strat = strat
		self.fitness_criterion = fitness_criterion
		self.simulator = simulator

		self.population_size = population_size
		self.num_generations = num_generations
		self.k = k
		self.p_cross = p_cross
		self.p_mut = p_mut

		self.population = None

	def fitness(self, start_idx, end_idx, individual, **kwargs):
		'''
		Determine fitness of a strategy.
		'''
		
		# Run a simulation
		strat = [individual, self.strat[1]]
		self.simulator.simulate(start_idx, end_idx, strat, **kwargs)
		self.simulator.end(end_idx)

		# Calculate a fitness score (the higher the better)
		portfolio = self.simulator.portfolio_history
		score = 0.0
		if self.fitness_criterion == 'min_vol':
			score = portfolio.volatility
			score = 1.0 / score
		elif self.fitness_criterion == 'max_sharpe':
			score = portfolio.sharpe_ratio
		elif self.fitness_criterion == 'max_sortino':
			score = portfolio.sortino_ratio
		else:
			score = portfolio.calmar_ratio

		# Reset the simulator
		self.simulator.reset()

		return score

	def crossover(self):
		'''
		Perform crossover on population with probabiltiy p_cross.
		'''
		
		# Shuffle the population
		random.shuffle(self.population)

		# Select two parents at a time for crossover
		for i in range(self.population_size, 2):

			# Create parents and children
			p1 = self.population[i]
			p2 = self.population[i+1]
			c1, c2 = p1.copy(), p2.copy()

			# Perform crossover with probabiltiy p_cross
			rand = random.random()
			if rand <= self.p_cross:
				# Select crossover point not at the end of a chromosome
				cross_pt = random.randint(1, len(p1)-1)
				# Crossover
				c1 = p1[:cross_pt] + p2[cross_pt:]
				c2 = p2[:cross_pt] + p1[cross_pt:]

			# Replace parents with children
			self.population[i] = c1
			self.population[i+1] = c2

	def mutation(self):
		'''
		Perform mutation on population with probability p_mut
		'''

		parameters = self.strat[0]
		
		# Perform mutation for each individual
		for i in range(self.population_size):

			# Perform mutation on each gene with probability p_mut
			for j in range(len(parameters)):

				rand = random.random()
				if rand <= self.p_mut:
					self.population[i][j] = random.choice(parameters[j])

	def tournament_selection(self, start_idx, end_idx, **kwargs):
		'''
		Perform tournament selection on k randomly-selected individuals.
		'''
		
		# Evaluate fitness of each individual
		population_with_fitness = []
		for i in range(self.population_size):
			print('\n\tSimulating portfolio {0}'.format(i+1))
			score = self.fitness(start_idx, end_idx, self.population[i])
			individual_with_fitness = [self.population[i], score]
			population_with_fitness.append(individual_with_fitness)

		# Perform tournament selection until we get a full population
		fittest_chromosomes = []
		for i in range(self.population_size):

			# Randomly sample population and select fittest individual
			sample = random.choices(population_with_fitness, k=self.k)
			best = None
			for j in range(len(sample)):
				if best == None or sample[j][1] > best[1]:
					best = sample[j]
			fittest_chromosomes.append(best[0])

		# Set the new population
		self.population = fittest_chromosomes

	def create_population(self):
		'''
		Create the initial population.
		'''
		
		self.population = []
		for i in range(self.population_size):
			individual = []
			for j in range(len(self.strat[0])): # Iterate over each strategy parameter
				individual.append(random.choice(self.strat[0][j]))
			self.population.append(individual)

	def optimize(self, start_idx, end_idx, **kwargs):
		'''
		For a given set of parameters, perform a genetic algorithm over several generations.
		Inputs:
    	- **kwargs: sector constraints for mean-variance portfolio construction (if any)
		'''

		# Create the population of potential strategies
		self.create_population()

		# Perform genetic algorithm
		for i in range(self.num_generations-1):
			print('\nGeneration {0}'.format(i+1))
			begin_time = time.time()
			self.tournament_selection(start_idx, end_idx, **kwargs)
			self.crossover()
			self.mutation()
			end_time = time.time()
			print('\n\tGeneration time: {0} seconds'.format(round(end_time-begin_time, 2)))

		# Select best individual
		best_strat = None
		best_fitness = None
		print('\nGeneration {0}'.format(self.num_generations))
		begin_time = time.time()
		for i in range(self.population_size):
			print('\n\tSimulating portfolio {0}'.format(i+1))
			score = self.fitness(start_idx, end_idx, self.population[i])
			if best_strat == None or score > best_fitness:
				best_strat = self.population[i]
				best_fitness = score
		end_time = time.time()
		print('\n\tGeneration time: {0} seconds'.format(round(end_time-begin_time, 2)))

		return best_strat


		















































