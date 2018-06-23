from __future__ import print_function, division

import array
import random
import sys

import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
import numpy as np
from deap import base
from deap import creator
from deap import tools

NGEN = 1000
NPOP = 80
CXPB = 0.8
MUTPB = 0.05
NDIM = 16


class Problem(object):
    """
    Reads and parses Open Shop Scheduling Problem
    """

    def __init__(self, filename, instance):
        self.filename = filename
        self.instance = instance
        self.numberOfJobs = 0
        self.numberOfMachines = 0
        self.processing_times = []
        self.operation_numbers_dictionary = {}
        self.dimension = 0
        self.machineOrder = []
        self.parse_problem()

    def parse_problem(self):
        """Parse the kth instance of a Taillard problem file

            The Taillard problem files are a standard benchmark set for the problem
            of flow shop scheduling. They can be found online at the following address:
            - http://mistic.heig-vd.ch/taillard/problemes.dir/ordonnancement.dir/ordonnancement.html"""

        with open(self.filename, 'r') as f:
            # Identify the string that separates instances
            problem_line = '/number of jobs, number of machines, time seed, machine seed, upper bound, lower bound :/'

            # Strip spaces and newline characters from every line
            lines = map(str.strip, f.readlines())

            # We prep the first line for later
            lines[0] = '/' + lines[0]

            # We also know '/' does not appear in the files, so we can use it as
            #  a separator to find the right lines for the kth problem instance
            try:
                proctimes = '/'.join(lines).split(problem_line)[self.instance].split('/machines')[0].split('/')[2:]
                machines = '/'.join(lines).split(problem_line)[self.instance].split('/machines')[1].split('/')[1:]
            except IndexError:
                max_instances = len('/'.join(lines).split(problem_line)) - 1
                print("\nError: Instance must be within 1 and %d\n" % max_instances)
                sys.exit(0)

            # Split every line based on spaces and convert each item to an int
            self.processing_times = [map(int, line.split()) for line in proctimes]

            self.numberOfJobs = len(self.processing_times)
            self.numberOfMachines = len(self.processing_times[0])
            self.dimension = self.numberOfJobs * self.numberOfMachines

            self.operation_numbers_dictionary = {i: (i % self.numberOfMachines, i // self.numberOfMachines)
                                                 for i in range(self.dimension)}

            self.machineOrder = [map(int, line.split()) for line in machines]

            self.sort_problem()

    def sort_problem(self):
        new_ptimes = np.zeros((self.numberOfJobs, self.numberOfMachines), dtype=np.int16)
        for i, job in enumerate(self.processing_times):
            newlist = sorted(zip(self.machineOrder[i], job))
            for j, ptime in enumerate(newlist):
                new_ptimes[i, j] = ptime[1]
        self.processing_times = new_ptimes.flatten()


class GA(object):
    def __init__(self, filename, instance):
        self.pr = Problem(filename, instance)
        self.NGEN = 1000
        self.NPOP = 80
        self.CXPB = 0.8
        self.MUTPB = 0.05
        self.NDIM = self.pr.dimension

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


filename = 'instances/Openshop/tai4_4.txt'
instance_number = 1
processing_times, machines = parse_problem(filename, instance_number)  # a list of [job number] [machine number]
numberOfJobs = len(processing_times)
numberOfMachines = len(processing_times[0])
NDIM = numberOfMachines * numberOfJobs

new_ptimes = np.zeros((numberOfJobs, numberOfMachines), dtype=np.int16)
for i, job in enumerate(processing_times):
    newlist = sorted(zip(machines[i], job))
    for j, ptime in enumerate(newlist):
        new_ptimes[i, j] = ptime[1]

# processing_times = [item for sublist in new_ptimes for item in sublist]
processing_times = new_ptimes.flatten()
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


# def check_intersection(time_interval_list, proc_time, current_machine_available_time):
#     while True:
#         if len(time_interval_list) != 0:
#             for times in time_interval_list:
#                 intersection = min(current_machine_available_time + proc_time, times[1]) - max(
#                     current_machine_available_time, times[0])
#                 if intersection > 0:
#                     current_machine_available_time = times[1]
#                     f_intersection = True
#                     break
#             else:
#                 f_intersection = False
#         else:
#             break
#
#         if not f_intersection:
#             break
#     return current_machine_available_time


# def gannt_chart_timegap(schedule):
#     """Compiles a scheduling on the machines given a permutation of jobs
#     with checking time gaps"""
#
#     flag_print = False
#
#     # Note that using [[]] * m would be incorrect, as it would simply
#     # copy the same list m times (as opposed to creating m distinct lists).
#     gantt_chart = [[] for _ in range(numberOfMachines)]
#
#     for operation in schedule:
#         machine_number = operation_numbers_dictionary[operation][0]
#         job_number = operation_numbers_dictionary[operation][1]
#         proc_time = processing_times[operation]
#
#         # check if this job is being processed in any other machine
#         completion_time_list = []
#         time_interval_list = []
#         for machine in range(numberOfMachines):
#             # dont check the machine to be scheduled since one job can be scheduled only once.
#             # Check other machines
#             if machine != machine_number:
#                 # check if the other machines had operations scheduled before
#                 if len(gantt_chart[machine]) != 0:
#                     for j in range(len(gantt_chart[machine])):
#                         # check the  job numbers on other machines
#                         # and determine if the machine processed an operation of the job
#                         # to be scheduled now
#                         if gantt_chart[machine][j][0] == job_number:
#                             # put completion times of the job on other machines into a list
#                             s_time = gantt_chart[machine][j][1]  # start time of the job on the other machine
#                             c_time = gantt_chart[machine][j][-1]  # completion time of the job on other machine
#                             completion_time_list.append(c_time)
#                             time_interval_list.append((s_time, c_time))
#                             time_interval_list.sort(key=lambda x: x[0])
#
#         # determine the maximum completion time for this job on other machines
#         if len(completion_time_list) != 0:
#             other_machine_ending_time = max(completion_time_list)
#
#         else:
#             # this job has no previous operation
#             other_machine_ending_time = 0
#
#         # determine the completion time of the last operation (available time) on the required machine
#         num_of_jobs_on_current_machine = len(gantt_chart[machine_number])
#         gaps = []
#         if num_of_jobs_on_current_machine == 0:
#             current_machine_available_time = 0
#             current_machine_available_time = check_intersection(time_interval_list, proc_time,
#                                                                 current_machine_available_time)
#         else:
#             # find gaps and put them in a list
#             for i in range(num_of_jobs_on_current_machine):
#                 if i == 0:
#                     first_job_start_time = gantt_chart[machine_number][i][1] - 0
#                     if first_job_start_time > 0:
#                         gaps.append((0, first_job_start_time))
#                 else:
#                     following_job_start_time = gantt_chart[machine_number][i][1]
#                     previous_job_end_time = gantt_chart[machine_number][i - 1][3]
#                     time_gap = following_job_start_time - previous_job_end_time
#                     if time_gap > 0:
#                         gaps.append((previous_job_end_time, following_job_start_time))
#
#             if len(gaps) == 0:
#                 # there are no gaps so the current available time is the last operation completion time
#                 current_machine_available_time = gantt_chart[machine_number][-1][-1]
#                 # check if there are no operation of the job on other machines and if there is an
#                 # intersection delay the current machine available time
#                 current_machine_available_time = check_intersection(time_interval_list, proc_time,
#                                                                     current_machine_available_time)
#             else:  # there are gaps
#                 flag_break_loop = False
#                 for times in gaps:
#                     # check if the operation processing time fits in the time gaps on this machine
#                     if times[1] - times[0] >= proc_time:
#                         # If operation processing time fits the time gap, then current machine is available
#                         # at the start of the time gap so set it with times[0]
#                         current_machine_available_time = times[0]
#                         if len(time_interval_list) != 0:
#                             # check if this gap intersects with other machines
#                             for times in time_interval_list:
#                                 intersection = min(current_machine_available_time + proc_time, times[1]) - max(
#                                     current_machine_available_time, times[0])
#                                 if intersection > 0:
#                                     break  # there is an intersection so break the loop
#                             else:
#                                 # te job fits the time gap and it doesnt have intersection with other
#                                 # operations on other machines therefore schedule it in the time gap
#                                 #current_machine_available_time = times[0]
#                                 flag_break_loop = True
#                         if flag_break_loop:
#                             break
#                 else:
#                     current_machine_available_time = gantt_chart[machine_number][-1][-1]
#                     # check if there are no operation of the job on other machines and if there is an
#                     # intersection delay the current machine available time
#                     current_machine_available_time = check_intersection(time_interval_list, proc_time,
#                                                                         current_machine_available_time)
#
#         completion_time = current_machine_available_time + proc_time
#         gantt_chart[machine_number].append((job_number, current_machine_available_time,
#                                             proc_time, completion_time))
#         gantt_chart[machine_number]= sorted(gantt_chart[machine_number], key= lambda x : x[1])
#     return gantt_chart


def gannt_chart(schedule):
    """Compiles a scheduling on the machines given a permutation of jobs 
    with no time gap checking"""

    fForceOrder = True

    # Note that using [[]] * m would be incorrect, as it would simply
    # copy the same list m times (as opposed to creating m distinct lists).

    gantt_chart = [[] for _ in range(numberOfMachines)]

    for operation in schedule:
        machine_number = operation_numbers_dictionary[operation][0]
        job_number = operation_numbers_dictionary[operation][1]
        proc_time = processing_times[operation]

        # determine the processing times of the job on other machines

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

                            time_interval_list.append((s_time, c_time))
                            time_interval_list.sort(key=lambda x: x[0])  # sort the list according to start time

        # determine the completion time of the last operation (available time) on the required machine
        num_of_jobs_on_current_machine = len(gantt_chart[machine_number])
        if num_of_jobs_on_current_machine == 0:
            current_machine_available_time = 0
        else:  # buradan emin degilim
            current_machine_available_time = gantt_chart[machine_number][-1][-1]

        if not fForceOrder:
            while True:
                if len(time_interval_list) != 0:
                    for times in time_interval_list:
                        # intersection1 = range(max(current_machine_available_time, times[0]),
                        #                      min(current_machine_available_time + proc_time, times[1]))
                        intersection = min(current_machine_available_time + proc_time, times[1]) - max(
                            current_machine_available_time, times[0])
                        if intersection > 0:
                            current_machine_available_time = times[1]
                            f_intersection = True
                            break
                    else:
                        f_intersection = False
                else:
                    break

                if not f_intersection:
                    break
            time_to_schedule = current_machine_available_time
        else:
            if len(time_interval_list) != 0:
                previous_completion_times = [i[-1] for i in time_interval_list]
                max_prev_ctimes = max(previous_completion_times)
                time_to_schedule = max(max_prev_ctimes, current_machine_available_time)
            else:
                time_to_schedule = current_machine_available_time

        completion_time = time_to_schedule + proc_time
        gantt_chart[machine_number].append((job_number, time_to_schedule,
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


def plot_gannt(machine_times, ms):
    """
    Plots the gannt chart of the given gannt chart data structure
    :param machine_times: gannt chart data structure
    :param ms: makespan value
    :return: None 
    """
    fig, ax = plt.subplots()
    facecolors = ('blue', 'red', 'yellow', 'green', 'grey', 'azure', 'plum',
                  'wheat', 'brown', 'chocolate', 'coral', 'cyan', 'darkblue',
                  'gold', 'khaki', 'lavender', 'lime', 'magenta', 'orange',
                  'pink')
    bar_start = 10
    bar_width = 9
    increment = 10
    for i in range(numberOfMachines):
        for j in range(numberOfJobs):
            datalist = [machine_times[i][j][1:3]]
            ax.broken_barh(datalist, (bar_start, bar_width),
                           facecolors=facecolors[machine_times[i][j][0]])
        bar_start += increment

    ax.set_ylim(5, 115)
    ax.set_xlim(0, ms)
    ytickpos = range(15, 85, 10)
    ax.set_yticks(ytickpos)
    yticklabels = ['Machine ' + str(i + 1) for i in range(numberOfMachines)]
    ax.set_yticklabels(yticklabels)
    ax.grid(True)

    fakeredbar = mpatch.Rectangle((0, 0), 1, 1, fc="r")
    fakebluebar = mpatch.Rectangle((0, 0), 1, 1, fc="b")
    fakeyellowbar = mpatch.Rectangle((0, 0), 1, 1, fc="y")
    fakegreenbar = mpatch.Rectangle((0, 0), 1, 1, fc="green")
    fakegreybar = mpatch.Rectangle((0, 0), 1, 1, fc="grey")
    fakeazurebar = mpatch.Rectangle((0, 0), 1, 1, fc='azure')
    fakeplumbar = mpatch.Rectangle((0, 0), 1, 1, fc='plum')

    plt.legend([fakebluebar, fakeredbar, fakeyellowbar, fakegreenbar, fakegreybar, fakeazurebar, fakeplumbar],
               ['Job1', 'Job2', 'Job3', 'Job4', 'Job5', 'Job6', 'Job7'])
    plt.show()


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
    # random.seed(8322)
    pr = Problem('instances/Openshop/tai4_4.txt', 1)
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
        # pop[:] = tools.selBest(pop, 1) + tools.selBest(offspring, NPOP - 1)  # use elitism
        pop[:] = offspring  # replace population with new offspring

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
    print(hof.items[0], " ", hof[0].fitness.values[0])
    best_schedule = hof.items[0]
    best_makespan = hof.items[0].fitness.values[0]
    best_gannt_chart = gannt_chart(best_schedule)
    for i in range(numberOfMachines):
        print(best_gannt_chart[i])
    plot_gannt(best_gannt_chart, best_makespan)


if __name__ == "__main__":
    main()
