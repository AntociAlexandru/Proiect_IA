##########################################################################
#                                                                        #
#  Copyright:   (c) 2024, Florin Leon                                    #
#  E-mail:      florin.leon@academic.tuiasi.ro                           #
#  Website:     http://florinleon.byethost24.com/lab_ia.html             #
#  Description: Evolutionary Algorithms                                  #
#               (Artificial Intelligence lab 8)                          #
#                                                                        #
#  This code and information is provided "as is" without warranty of     #
#  any kind, either expressed or implied, including but not limited      #
#  to the implied warranties of merchantability or fitness for a         #
#  particular purpose. You are free to use this source code in your      #
#  applications as long as the original copyright notice is included.    #
#                                                                        #
##########################################################################
import random
import math

# Interface for optimization problems
class IOptimizationProblem:
    def compute_fitness(self, chromosome):
        raise NotImplementedError("This method needs to be implemented by a subclass")

    def make_chromosome(self):
        raise NotImplementedError("This method needs to be implemented by a subclass")


class Chromosome:
    def __init__(self, no_genes, min_values, max_values):
        self.no_genes = no_genes
        self.genes = [0.0] * no_genes
        self.min_values = list(min_values)
        self.max_values = list(max_values)
        self.fitness = 0.0
        self._initialize_genes()

    def _initialize_genes(self):
        for i in range(self.no_genes):
            self.genes[i] = self.min_values[i] + random.random() * (self.max_values[i] - self.min_values[i])

    def copy(self):
        clone = Chromosome(self.no_genes, self.min_values, self.max_values)
        clone.genes = list(self.genes)
        clone.fitness = self.fitness
        return clone


class Selection:
    @staticmethod
    def tournament(population):
        rand1 = random.randint(0, len(population) - 1)
        rand2 = random.randint(0, len(population) - 1)
        return population[rand1] if population[rand1].fitness >= population[rand2].fitness else population[rand2]

    @staticmethod
    def get_best(population):
        return max(population, key=lambda c: c.fitness).copy()


class Crossover:
    @staticmethod
    def arithmetic(mother, father, rate):
        if random.random() >= rate:
            return mother.copy() if random.random() < 0.5 else father.copy()
        else:
            a = random.uniform(-0.25, 1.25)
            child_genes = [(a * m + (1 - a) * f) for m, f in zip(mother.genes, father.genes)]
            child = Chromosome(mother.no_genes, mother.min_values, mother.max_values)
            child.genes = child_genes
            return child


class Mutation:
    @staticmethod
    def reset(child, rate):
        for i in range(child.no_genes):
            if random.random() < rate:
                child.genes[i] = child.min_values[i] + random.random() * (child.max_values[i] - child.min_values[i])


class EvolutionaryAlgorithm:
    def solve(self, problem, population_size, max_generations, crossover_rate, mutation_rate):
        # Initialize population
        population = [problem.make_chromosome() for _ in range(population_size)]
        for individual in population:
            problem.compute_fitness(individual)

        for gen in range(max_generations):
            new_population = [Selection.get_best(population)]  # Elitism

            for _ in range(1, population_size):
                # Select parents
                father = Selection.tournament(population)
                mother = Selection.tournament(population)
                # Generate child through crossover
                child = Crossover.arithmetic(mother, father, crossover_rate)
                # Mutate child
                Mutation.reset(child, mutation_rate)
                # Compute fitness of child
                problem.compute_fitness(child)
                new_population.append(child)

            population = new_population

        return Selection.get_best(population)