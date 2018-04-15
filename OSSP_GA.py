from __future__ import print_function, division
from deap import base
from deap import tools
from deap import creator
from collections import deque
import array

import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.markers import CARETDOWN
import numpy.polynomial.polynomial as poly
import math

NGEN = 2000
NPOP = 80
CXPB = 0.8
MUTPB = 0.2
NDIV = 70
fApplyDiversity = False

# Read processing times from the file
def parse_problem(filename, k=1):
    """Parse the kth instance of a Taillard problem file

    The Taillard problem files are a standard benchmark set for the problem
    of flow shop scheduling. They can be found online at the following address:
    - http://mistic.heig-vd.ch/taillard/problemes.dir/ordonnancement.dir/ordonnancement.html"""

    with open(filename, 'r') as f:
        # Identify the string that separates instances
        problem_line = '/number of jobs, number of machines, initial seed, upper bound and lower bound :/'

        # Strip spaces and newline characters from every line
        lines = map(str.strip, f.readlines())

        # We prep the first line for later
        lines[0] = '/' + lines[0]

        # We also know '/' does not appear in the files, so we can use it as
        #  a separator to find the right lines for the kth problem instance
        try:
            lines = '/'.join(lines).split(problem_line)[k].split('/')[2:]
        except IndexError:
            max_instances = len('/'.join(lines).split(problem_line)) - 1
            print("\nError: Instance must be within 1 and %d\n" % max_instances)
            sys.exit(0)

        # Split every line based on spaces and convert each item to an int
        data = [map(int, line.split()) for line in lines]

    # We return the zipped data to rotate the rows and columns, making each
    #  item in data the durations of tasks for a particular job
    return zip(*data)
filename = 'instances/tai20_5.txt'
processing_times = parse_problem(filename, 1)
numberOfJobs = len(processing_times)
numberOfMachines = len(processing_times[0])
NDIM = numberOfMachines * numberOfJobs
print(NDIM)
# define the problem as an Minimization or Maximization by defining the weights
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# create a container to hold individuals which have an attribute named "fitness"
creator.create("Individual", array.array, typecode='B', fitness=creator.FitnessMin)

toolbox = base.Toolbox()  # get toolbox class from base module of deap
toolbox.register("attribute", random.sample, range(NDIM), NDIM)  # register attribute method to toolbox
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attribute)  # register individual method
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # register population method to toolbox


# Define Objective Function
def make_span(individual):
   pass


def fitnessdiversity(population):
    """
    Calculates and returns the number of unique fitness values 
    """
    fitnessvals = [int(individual.fitness.values[0]) for individual in population]
    fitset = set(fitnessvals)
    return len(fitset)


def solutiondiversity(population):
    solutions = [tuple(individual) for individual in population]
    unique_solution_set = set(solutions)
    unique_pop = map(creator.Individual, list(unique_solution_set))
    fitnesses = map(toolbox.evaluate, unique_pop)
    for ind, fit in zip(unique_pop, fitnesses):
        ind.fitness.values = fit
    return len(unique_solution_set), unique_pop


def swap(individual):
    idx1, idx2 = random.sample(range(NDIM), 2)
    a, b = individual.index(idx1), individual.index(idx2)
    individual[b], individual[a] = individual[a], individual[b]
    return individual


def create_offspring(pop):
    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))

    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))  # clone offsprings

    # Apply crossover on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # Apply mutation on the offspring
    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    return offspring


# register objective function with the "evaluate" alias to the toolbox
toolbox.register("evaluate", make_span)

# register crossover function with the "mate" alias to the toolbox
toolbox.register("mate", tools.cxOrdered)

# register mutation function with the "mutate" alias to the toolbox
toolbox.register("mutate", swap)

# register selection function with the "select" alias to the toolbox
toolbox.register("select", tools.selTournament, tournsize=2)


def main():
    # random.seed(169)
    pop = toolbox.population(n=NPOP)
    print(pop)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    # stats.register("diversity", fitnessdiversity)

    logbook = tools.Logbook()

    # assign logbook headers
    logbook.header = "gen", "evals", "min", "max", "avg", "std", 'fitdiv', 'soldiv', 'distance'

    # populasyondaki her bir bireyin degerini hesapla ve
    # o bireyin fitness degerini guncelle
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), fitdiv=fitnessdiversity(pop),
                   soldiv=solutiondiversity(pop)[0], distance=NDIM, **record)
    print(logbook.stream)
    hof.update(pop)

    cbestfitness = hof.items[0].fitness.values[0]

    boolidx = np.empty((NGEN,), dtype=np.bool_)  # create an empty bool array to hold progress flags

    for gen in xrange(1, NGEN):

        if fApplyDiversity:
            numberOfUniqueSolutions, new_pop = solutiondiversity(pop)
            if numberOfUniqueSolutions < NDIV:
                # print(tools.selBest(new_pop, 1)[0].fitness.values[0])
                while len(new_pop) != NPOP:
                    new_individual = toolbox.individual()
                    if new_individual not in new_pop:
                        fitness = toolbox.evaluate(new_individual)
                        new_individual.fitness.values = fitness
                        new_pop.append(new_individual)
                # offspring = create_offspring(new_pop)
                # pop[:] = tools.selBest(new_pop, 1) + tools.selBest(offspring, NPOP - 1)
                pop[:] = new_pop
            else:
                offspring = create_offspring(pop)
                pop[:] = tools.selBest(pop, 1) + tools.selBest(offspring, NPOP - 1)
        else:
            offspring = create_offspring(pop)
            pop[:] = tools.selBest(pop, 1) + tools.selBest(offspring, NPOP - 1)

        hof.update(pop)
        bestInPop, dist = toolbox.circshift([i + 1 for i in hof[0]])

        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(pop), fitdiv=fitnessdiversity(pop),
                       soldiv=solutiondiversity(pop)[0], distance=dist, **record)
        print(logbook.stream)

        # if the best individual in new population has a better value than
        # the previous best then update the current best fitness and mark the
        # progress
        if hof.items[0].fitness.values[0] < cbestfitness:
            cbestfitness = hof.items[0].fitness.values[0]
            boolidx[gen] = True  # mark the generation as a progress
        else:
            boolidx[gen] = False  # no progress

    print("-- End of (successful) evolution --")

    best_ind, dist = toolbox.circshift([i + 1 for i in hof[0]])

    print("Best {:>}, {:>}".format(best_ind, hof[0].fitness.values[0]))

    print('Opt. {:>}, {:>}'.format(bestsolution, optimalTour))
    gen, evals, avg, minval, fitness_diversity, std, solution_diversity, Levdist = \
        logbook.select('gen', 'evals', 'avg', 'min', 'fitdiv', 'std', 'soldiv', 'distance')

    progressArray = [element for i, element in enumerate(minval) if boolidx[i]]
    progressArray.insert(0, minval[0])  # put first value to the progress array


    genArray = [0] + [element for i, element in enumerate(range(0, NGEN)) if boolidx[i]]

   
if __name__ == "__main__":
    main()
