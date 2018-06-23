from __future__ import print_function, division

import array
import random
import sys

import numpy as np
from deap import base
from deap import creator
from deap import tools

NGEN = 500
NPOP = 200
CXPB = 0.6
MUTPB = 0.1


# Read processing times from the file
def parse_problem(filename, k=1):
    """Parse the kth instance of a Taillard problem file

    The Taillard problem files are a standard benchmark set for the problem
    of flow shop scheduling. They can be found online at the following address:
    - http://mistic.heig-vd.ch/taillard/problemes.dir/ordonnancement.dir/ordonnancement.html"""

    with open(filename, 'r') as f:
        # Identify the string that separates instances
        problem_line = '/number of jobs, number of machines, time seed, machine seed, upper bound, lower bound :/'

        # Strip spaces and newline characters from every line
        lines = map(str.strip, f.readlines())

        # We prep the first line for later
        lines[0] = '/' + lines[0]

        # We also know '/' does not appear in the files, so we can use it as
        #  a separator to find the right lines for the kth problem instance
        try:
            proctimes = '/'.join(lines).split(problem_line)[k].split('/machines')[0].split('/')[2:]
            machines = '/'.join(lines).split(problem_line)[k].split('/machines')[1].split('/')[1:]
        except IndexError:
            max_instances = len('/'.join(lines).split(problem_line)) - 1
            print("\nError: Instance must be within 1 and %d\n" % max_instances)
            sys.exit(0)

        # Split every line based on spaces and convert each item to an int
        data = [map(int, line.split()) for line in proctimes]

        machines = [map(int, line.split()) for line in machines]

    # We return the zipped data to rotate the rows and columns, making each
    # item in data the durations of tasks for a particular job
    return data, machines


filename = 'instances/Openshop/tai7_7.txt'
instance_number = 2
processing_times, machines = parse_problem(filename, instance_number)  # a list of [job number] [machine number]
numberOfJobs = len(processing_times)
numberOfMachines = len(processing_times[0])
NDIM = numberOfMachines * numberOfJobs

new_ptimes = []
for idx, job in enumerate(processing_times):
    newlist = sorted(zip(machines[idx], job))
    ptimes_inorder = [element[1] for element in newlist]
    new_ptimes.append(ptimes_inorder)

processing_times = [item for sublist in new_ptimes for item in sublist]
operation_numbers_dictionary = {i: (i % numberOfMachines, i // numberOfMachines)
                                for i in range(NDIM)}

# define the problem as an Minimization or Maximization by defining the weights
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# create a container to hold individuals which have an attribute named "fitness"
creator.create("Individual", array.array, typecode='B', fitness=creator.FitnessMin)

toolbox = base.Toolbox()  # get toolbox class from base module of deap
toolbox.register("attribute", random.sample, range(NDIM), NDIM)  # register attribute method to toolbox
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attribute)  # register individual method
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # register population method to toolbox


# noinspection PyTypeChecker,PyTypeChecker
def gannt_chart(schedule):
    """Compiles a scheduling on the machines given a permutation of jobs"""

    flag_print = False

    # Note that using [[]] * m would be incorrect, as it would simply
    # copy the same list m times (as opposed to creating m distinct lists).
    gantt_chart = [[] for _ in range(numberOfMachines)]

    for operation in schedule:
        machine_number = operation_numbers_dictionary[operation][0]
        job_number = operation_numbers_dictionary[operation][1]
        proc_time = processing_times[operation]

        # determine the processing times of the job on other machines
        completion_time_list = []
        time_interval_list = []
        for machine in range(numberOfMachines):
            # dont check the machine to be scheduled since one job can be scheduled only once.
            # Check other machines
            if machine != machine_number:
                # check if the other machines had operations scheduled before
                if len(gantt_chart[machine]) != 0:
                    for j in range(len(gantt_chart[machine])):
                        # check the  job numbers on other machines
                        # and determine if the machine processed an operation of the job
                        # to be scheduled now
                        if gantt_chart[machine][j][0] == job_number:
                            # put completion times of the job on other machines into a list
                            s_time = gantt_chart[machine][j][1]  # start time of the job on the other machine
                            c_time = gantt_chart[machine][j][-1]  # completion time of the job on other machine
                            completion_time_list.append(c_time)
                            time_interval_list.append((s_time, c_time))
                            time_interval_list.sort(key=lambda x: x[0])  # sort the list according to start time

        # determine the completion time of the last operation (available time) on the required machine
        num_of_jobs_on_current_machine = len(gantt_chart[machine_number])
        if num_of_jobs_on_current_machine == 0:
            current_machine_available_time = 0
        else:  # buradan emin degilim
            current_machine_available_time = gantt_chart[machine_number][-1][-1]

        while True:
            if len(time_interval_list) != 0:
                for times in time_interval_list:
                    intersection = range(max(current_machine_available_time, times[0]),
                                         min(current_machine_available_time + proc_time, times[1]))

                    if len(intersection) > 0:
                        current_machine_available_time = times[1]
                        f_intersection = True
                        break
                else:
                    f_intersection = False
            else:
                break

            if not f_intersection:
                break

        completion_time = current_machine_available_time + proc_time
        gantt_chart[machine_number].append((job_number, current_machine_available_time,
                                            proc_time, completion_time))
    return gantt_chart


# Define Objective Function
def makespan(schedule):
    gannt_chrt = gannt_chart(schedule)
    ctimes = []
    for machine in range(numberOfMachines):
        ctimes.append(gannt_chrt[machine][-1][-1])
    make_span = max(ctimes)
    return make_span,  # return a tuple for compatibility


# def plot_gannt(machine_times, ms):
#     """
#     Plots the gannt chart of the given gannt chart data structure
#     :param machine_times: gannt chart data structure
#     :param ms: makespan value
#     :return: None
#     """
#     fig, ax = plt.subplots()
#     facecolors = ('blue', 'red', 'yellow', 'green', 'grey', 'azure', 'plum',
#                   'wheat', 'brown', 'chocolate', 'coral', 'cyan', 'darkblue',
#                   'gold', 'khaki', 'lavender', 'lime', 'magenta', 'orange',
#                   'pink')
#     bar_start = 10
#     bar_width = 9
#     increment = 10
#     for i in range(numberOfMachines):
#         for j in range(numberOfJobs):
#             datalist = [machine_times[i][j][1:3]]
#             ax.broken_barh(datalist, (bar_start, bar_width),
#                            facecolors=facecolors[machine_times[i][j][0]])
#         bar_start += increment
#
#     ax.set_ylim(5, 115)
#     ax.set_xlim(0, ms)
#     ytickpos = range(15,85,10)
#     ax.set_yticks(ytickpos)
#     yticklabels = ['Machine ' + str(i + 1) for i in range(numberOfMachines)]
#     ax.set_yticklabels(yticklabels)
#     ax.grid(True)
#
#     fakeredbar = mpatch.Rectangle((0, 0), 1, 1, fc="r")
#     fakebluebar = mpatch.Rectangle((0, 0), 1, 1, fc="b")
#     fakeyellowbar = mpatch.Rectangle((0, 0), 1, 1, fc="y")
#     fakegreenbar = mpatch.Rectangle((0, 0), 1, 1, fc="green")
#     fakegreybar = mpatch.Rectangle((0, 0), 1, 1, fc="grey")
#     fakeazurebar = mpatch.Rectangle((0, 0), 1, 1, fc='azure')
#     fakeplumbar = mpatch.Rectangle((0, 0), 1, 1, fc='plum')
#
#     plt.legend([fakebluebar, fakeredbar, fakeyellowbar, fakegreenbar, fakegreybar, fakeazurebar, fakeplumbar],
#                ['Job1', 'Job2', 'Job3', 'Job4', 'Job5', 'Job6', 'Job7'])
#     plt.show()


def swap(individual):
    idx1, idx2 = random.sample(range(NDIM), 2)
    a, b = individual.index(idx1), individual.index(idx2)
    individual[b], individual[a] = individual[a], individual[b]
    return individual


def evolve(pop):
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
toolbox.register("evaluate", makespan)

# register crossover function with the "mate" alias to the toolbox
toolbox.register("mate", tools.cxOrdered)

# register mutation function with the "mutate" alias to the toolbox
toolbox.register("mutate", swap)

# register selection function with the "select" alias to the toolbox
toolbox.register("select", tools.selTournament, tournsize=2)


def main():
    # random.seed(169)
    pop = toolbox.population(n=NPOP)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()

    # assign logbook headers
    logbook.header = "gen", "evals", "min", "max", "avg", "std"

    # populasyondaki her bir bireyin degerini hesapla ve
    # o bireyin fitness degerini guncelle
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)
    print(logbook.stream)
    hof.update(pop)

    cbestfitness = hof.items[0].fitness.values[0]

    boolidx = np.empty((NGEN,), dtype=np.bool_)  # create an empty bool array to hold progress flags

    for gen in range(1, NGEN):
        offspring = evolve(pop)
        pop[:] = tools.selBest(pop, 1) + tools.selBest(offspring, NPOP - 1)  # use elitism
        # pop[:] = offspring  # replace population with new offspring

        hof.update(pop)

        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(pop), **record)
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

    gen, evals, avg, minval, std = logbook.select('gen', 'evals', 'avg', 'min', 'std')
    # print(hof.items[0], " ", hof[0].fitness.values[0])
    best_schedule = hof.items[0]
    best_makespan = hof.items[0].fitness.values[0]
    best_gannt_chart = gannt_chart(best_schedule)
    for i in range(numberOfMachines):
        print(best_gannt_chart[i])
    print(best_makespan)



if __name__ == "__main__":
    main()
