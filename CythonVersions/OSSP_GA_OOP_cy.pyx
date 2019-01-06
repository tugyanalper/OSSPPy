from __future__ import print_function, division

import array
import random
import sys
from collections import deque

import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
import numpy as np
from deap import base
from deap import creator
from deap import tools
from simanneal import Annealer


class Problem(object):
    """
    Reads and parses Open Shop Scheduling Problem
    """

    def __init__(self, filename, instance):
        self.filename = filename
        self.instance = instance
        self.numberOfJobs = 0
        self.numberOfMachines = 0
        self.processing_times = None
        self.operation_numbers_dictionary = None
        self.dimension = 0
        self.machineOrder = None
        self.due_dates = None
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
            lines = list(map(str.strip, f.readlines()))

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
            self.processing_times = [list(map(int, line.split())) for line in proctimes]

            self.numberOfJobs = len(self.processing_times)
            self.numberOfMachines = len(self.processing_times[0])
            self.dimension = self.numberOfJobs * self.numberOfMachines

            self.operation_numbers_dictionary = {i: (i % self.numberOfMachines, i // self.numberOfMachines)
                                                 for i in range(self.dimension)}  # Operation : (Machine Nu., Job Nu.)

            self.machineOrder = [map(int, line.split()) for line in machines]

            self.sort_problem()

    def sort_problem(self):
        new_ptimes = np.zeros((self.numberOfJobs, self.numberOfMachines), dtype=np.int16)
        for i, job in enumerate(self.processing_times):
            newlist = sorted(zip(self.machineOrder[i], job))
            for j, ptime in enumerate(newlist):
                new_ptimes[i, j] = ptime[1]
        self.due_dates = np.sum(new_ptimes, axis=1)  # tight due dates
        self.processing_times = new_ptimes.flatten()


class SA(Annealer):
    def __init__(self, state, problem):
        self.problem = problem
        super(SA, self).__init__(state)  # important!

    def move(self):
        """Swaps two operations in the route."""
        a = random.randint(0, len(self.state) - 1)
        b = random.randint(0, len(self.state) - 1)
        self.state[a], self.state[b] = self.state[b], self.state[a]

    def energy(self):
        self.gannt_chrt = self.gannt_chart(self.state)
        ctimes = []
        for machine in range(self.problem.numberOfMachines):
            ctimes.append(self.gannt_chrt[machine][-1][-1])
        make_span = max(ctimes)
        return make_span

    def gannt_chart(self, state):
        """
        Compiles a scheduling on the machines given a permutation of jobs 
        with no time gap checking
        :return: gantt_chart list of list of tuples (job number, start time, processing time, completion time)
        """

        fForceOrder = True

        # Note that using [[]] * m would be incorrect, as it would simply
        # copy the same list m times (as opposed to creating m distinct lists).

        gantt_chart = [[] for _ in range(self.problem.numberOfMachines)]

        for operation in state:
            machine_number = self.problem.operation_numbers_dictionary[operation][0]
            job_number = self.problem.operation_numbers_dictionary[operation][1]
            proc_time = self.problem.processing_times[operation]

            # determine the processing times of the job on other machines

            time_interval_list = []
            for machine in range(self.problem.numberOfMachines):
                # dont check the machine to be scheduled since one job can be scheduled only once.
                # Check other machines if they have operations scheduled before
                if machine != machine_number and len(gantt_chart[machine]) != 0:
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


class OpenShopGA(object):
    """
    A class that solves Open Shop Scheduling Problem
    """

    def __init__(self, problem, objective='makespan', mutation='swap', crossover='one_point',
                 max_gen=1000, pop_size=80, cross_pb=0.8, mut_pb=0.05, fprint=False, strategy='normal',
                 fApplyDiversity=False, diversity_metric='distance', fApplySA=False, fForceOrder=False):
        self.problem = problem
        self.NGEN = max_gen
        self.NPOP = pop_size
        self.CXPB = cross_pb
        self.MUTPB = mut_pb
        self.objective = objective
        self.mutation = mutation
        self.crossover = crossover
        self.pop = None
        self.hof = None
        self.toolbox = None
        self.logbook = None
        self.stats = None
        self.due_dates = None
        self.area_ratio = None
        self.diversity = 1
        self.strategy = strategy
        self.fApplyDiversity = fApplyDiversity
        self.diversity_metric = diversity_metric
        self.flag_print = fprint
        self.NDIM = self.problem.dimension
        self.fApplySA = fApplySA
        self.fForceOrder = fForceOrder

        self.register_functions()  # register required functions
        self.generate_population()  # create the initial population
        self.init_stats()  # get statistics about the initial population

    # Objective Functions #
    def makespan(self, schedule):
        gannt_chrt = self.gannt_chart(schedule)
        ctimes = []
        for machine in range(self.problem.numberOfMachines):
            ctimes.append(gannt_chrt[machine][-1][-1])
        make_span = max(ctimes)
        return make_span,  # return a tuple for compatibility

    def sum_of_completion_times(self, schedule):
        gannt_chrt = self.gannt_chart(schedule)
        sum_of_ctimes = 0
        for machine in range(self.problem.numberOfMachines):
            sum_of_ctimes += gannt_chrt[machine][-1][1]
        return sum_of_ctimes,  # return a tuple for compatibility

    def total_tardiness(self, schedule):
        gannt_chrt = self.gannt_chart(schedule)

        job_times = [[] for _ in range(self.problem.numberOfJobs)]

        for machine in range(self.problem.numberOfMachines):
            gannt_chrt[machine].sort(key=lambda x: x[0])
            for idx, job in enumerate(gannt_chrt[machine]):
                job_times[idx].append((job[0], job[3]))

        ctimes = []
        for job in range(self.problem.numberOfJobs):
            job_times[job].sort(key=lambda x: x[1])
            ctimes.append(job_times[job][-1][-1])

        tardiness = 0
        for idx, due_date in enumerate(self.problem.due_dates):
            if ctimes[idx] > due_date:
                tardiness += ctimes[idx] - due_date

        return tardiness,

    # Crossover Functions #
    @staticmethod
    def one_point_crossover(child1, child2):
        """Executes a one point crossover on the input :term:`sequence` individuals.
            The two individuals are modified in place. The resulting individuals will
            respectively have the length of the other.

            :param child1: The first individual participating in the crossover.
            :param child2: The second individual participating in the crossover.
            :returns: A tuple of two individuals.

            This function uses the :func:`~random.randint` function from the
            python base :mod:`random` module.
            """
        size = min(len(child1), len(child2))
        cxpoint = random.randint(1, size - 1)
        # child1[cxpoint:], child2[cxpoint:] = child2[cxpoint:], child1[cxpoint:]
        of1 = child1[:cxpoint]
        of2 = child2[:cxpoint]

        temp1 = array.array('B', [element for element in child2 if element not in of1])
        temp2 = array.array('B', [element for element in child1 if element not in of2])

        of1 = of1 + temp1
        of2 = of2 + temp2

        child1[:], child2[:] = of1[:], of2[:]
        return child1, child2

    @staticmethod
    def two_point_crossover(child1, child2):
        """Executes a two-point crossover on the input :term:`sequence`
           individuals. The two individuals are modified in place and both keep
           their original length. 

           :param child1: The first individual participating in the crossover.
           :param child2: The second individual participating in the crossover.
           :returns: A tuple of two individuals.

           This function uses the :func:`~random.randint` function from the Python 
           base :mod:`random` module.
           """
        size = min(len(child1), len(child2))
        cxpoint1 = random.randint(1, size)
        cxpoint2 = random.randint(1, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else:  # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        ss1_of_ch1 = child1[:cxpoint1]  # substring 1 of child 1
        ss2_of_ch1 = child1[cxpoint1:cxpoint2]  # substring 2 of child 1
        ss3_of_ch1 = child1[cxpoint2:]  # substring 3 of child 1

        ss1_of_ch2 = child2[:cxpoint1]  # substring 1 of child 2
        ss2_of_ch2 = child2[cxpoint1:cxpoint2]  # substring 2 of child 2
        ss3_of_ch2 = child2[cxpoint2:]  # substring 3 of child 2

        checkarray1 = ss3_of_ch1 + ss1_of_ch1 + ss2_of_ch1
        checkarray2 = ss3_of_ch2 + ss1_of_ch2 + ss2_of_ch2

        orderedarray1 = array.array('B', [element for element in checkarray1 if element not in ss2_of_ch2])
        orderedarray2 = array.array('B', [element for element in checkarray2 if element not in ss2_of_ch1])

        new_ss3_of_ch2 = orderedarray1[:len(ss3_of_ch2)]
        new_ss1_of_ch2 = orderedarray1[len(ss3_of_ch2):]

        child2[:] = new_ss1_of_ch2 + ss2_of_ch2 + new_ss3_of_ch2

        new_ss3_of_ch1 = orderedarray2[:len(ss3_of_ch1)]
        new_ss1_of_ch1 = orderedarray2[len(ss3_of_ch1):]

        child1[:] = new_ss1_of_ch1 + ss2_of_ch1 + new_ss3_of_ch1

        return child1, child2

    def linear_order_crossover(self, child1, child2):
        size = min(len(child1), len(child2))
        cxpoint1 = random.randint(1, size)
        cxpoint2 = random.randint(1, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else:  # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        # ss1_of_ch1 = child1[:cxpoint1]  # substring 1 of child 1
        ss2_of_ch1 = child1[cxpoint1:cxpoint2]  # substring 2 of child 1
        # ss3_of_ch1 = child1[cxpoint2:]  # substring 3 of child 1

        # ss1_of_ch2 = child2[:cxpoint1]  # substring 1 of child 24987
        ss2_of_ch2 = child2[cxpoint1:cxpoint2]  # substring 2 of child 2
        # ss3_of_ch2 = child2[cxpoint2:]  # substring 3 of child 2

        offspring1 = self.toolbox.clone(child1)
        offspring1[cxpoint1:cxpoint2] = ss2_of_ch1

        index = 0
        for element in child2:
            if element not in ss2_of_ch1:
                if index < cxpoint1 or index >= cxpoint2:
                    offspring1[index] = element
                    index += 1
                else:
                    index = cxpoint2
                    offspring1[index] = element
                    index += 1

        offspring2 = self.toolbox.clone(child2)
        offspring2[cxpoint1:cxpoint2] = ss2_of_ch2
        child1 = offspring1
        del offspring1

        index = 0
        for element in child1:
            if element not in ss2_of_ch2:
                if index < cxpoint1 or index >= cxpoint2:
                    offspring2[index] = element
                    index += 1
                else:
                    index = cxpoint2
                    offspring2[index] = element
                    index += 1
        child2 = offspring2
        del offspring2
        return child1, child2

    def three_chromosome_juggling_forward(self, child1, child2, child3):
        offspring1 = self.toolbox.clone(child1)
        offspring2 = self.toolbox.clone(child2)
        c1 = self.toolbox.clone(child1)
        c2 = self.toolbox.clone(child2)
        c3 = self.toolbox.clone(child3)
        for i, operation in enumerate(c3):
            ops_tuple = self.problem.operation_numbers_dictionary[operation]
            if sum(ops_tuple) % 2 == 0:
                offspring1[i] = c2[0]
                val = c2[0]
                del c2[0]
                idx1 = c1.index(val)
                del c1[idx1]
            else:
                offspring1[i] = c1[0]
                val = c1[0]
                del c1[0]
                idx2 = c2.index(val)
                del c2[idx2]

        c1 = self.toolbox.clone(child1)
        c2 = self.toolbox.clone(child2)
        c3 = self.toolbox.clone(child3)
        for i, operation in enumerate(c1):
            ops_tuple = self.problem.operation_numbers_dictionary[operation]
            if sum(ops_tuple) % 2 == 0:
                offspring2[i] = c2[0]
                val = c2[0]
                del c2[0]
                idx1 = c3.index(val)
                del c3[idx1]
            else:
                offspring2[i] = c3[0]
                val = c3[0]
                del c3[0]
                idx2 = c2.index(val)
                del c2[idx2]

        child1 = offspring1
        child2 = offspring2
        return child1, child2, child3

    def gchart_crossover(self, child1, child2):
        # convert permutation schedules to machine ordered lists
        schedule1 = self.operation_scheduler(child1)
        schedule2 = self.operation_scheduler(child2)

        # create a list to hold the operations that intersect on both schedules
        holes = [[] for _ in range(self.problem.numberOfMachines)]
        intersection = []
        for i in range(self.problem.numberOfMachines):
            for j in range(self.problem.numberOfJobs):
                if schedule1[i][j] == schedule2[i][j]:
                    holes[i].append(schedule1[i][j])
                    intersection.append(schedule1[i][j])
                else:
                    # if no intersection then put a -1 for that place
                    holes[i].append(-1)

        # put jobs that do not intersect into leftover list
        leftover = set(child1).difference(set(intersection))

        # get the processing times of left over jobs and put them into "pt" list
        pt = [self.problem.processing_times[operation] for operation in leftover]

        # zip the leftover jobs and their corresponding processing times into "zpt" list
        zpt = zip(leftover, pt)

        # sort the zpt list in descending order according to the processing times
        zpt.sort(reverse=True, key=lambda x: x[1])

        return holes

    # Mutation Functions #
    def swap(self, individual):
        idx1, idx2 = random.sample(range(self.NDIM), 2)
        a, b = individual.index(idx1), individual.index(idx2)
        individual[b], individual[a] = individual[a], individual[b]
        return individual

    def insert(self, individual):
        idx1, idx2 = random.sample(range(self.NDIM), 2)  # select a random index
        element = individual[idx1]  # get the random operation based on the index
        del individual[idx1]
        individual.insert(idx2, element)
        return individual

    def shift(self, individual):
        shift_idx = random.sample(range(self.NDIM), 1)  # returns a list
        temp = deque(individual[:])
        temp.rotate(shift_idx[0])
        individual[:] = array.array('B', temp)[:]
        return individual

    @staticmethod
    def shuffle(individual):
        """
        The operations between the two randomly selected points are shuffled.
        :param individual: 
        :return: 
        """
        size = len(individual)
        point1 = random.randint(1, size)
        point2 = random.randint(1, size - 1)
        if point2 >= point1:
            point2 += 1
        else:  # Swap the two cx points
            point1, point2 = point2, point1

        scrambled_section = individual[point1:point2]
        random.shuffle(scrambled_section)

        individual[point1:point2] = scrambled_section[:]

        return individual

    @staticmethod
    def inversion(individual):
        """
        The operations between the two randomly selected points are shuffled.
        :param individual: 
        :return: 
        """
        size = len(individual)
        point1 = random.randint(1, size)
        point2 = random.randint(1, size - 1)
        if point2 >= point1:
            point2 += 1
        else:  # Swap the two cx points
            point1, point2 = point2, point1

        section = individual[point1:point2]
        section.reverse()

        individual[point1:point2] = section[:]

        return individual

    def slacktimemutation(self, schedule):
        gannt_chart = self.gannt_chart(schedule)
        # self.plot_gannt(schedule)
        # now find the operations in the gannt chart that
        # has idle time before it is scheduled. Gannt chart
        # is a list of lists.
        # (1: Operation Number, 2: Start Time,
        # 3: Processing Time, 4: Completion Time)
        jobs_with_slack = []
        for i in range(len(gannt_chart)):
            for j in range(self.problem.numberOfJobs - 1):

                # check the first operation's start time on each machine
                # if it is not zero than it has a gap
                if j == 0 and gannt_chart[i][j][1] != 0:
                    operation_number = i * self.problem.numberOfJobs + gannt_chart[i][j][0]
                    jobs_with_slack.append(operation_number)

                # subtract the start time of the next job from
                # the previous completion time
                time_gap = gannt_chart[i][j + 1][1] - gannt_chart[i][j][3]
                if time_gap > 0:
                    operation_number = i * self.problem.numberOfJobs + gannt_chart[i][j + 1][0]
                    jobs_with_slack.append(operation_number)

        if len(jobs_with_slack) >= 2:
            idx = random.sample(range(len(jobs_with_slack)), 1)[0]
            a = schedule.index(jobs_with_slack[idx])

            element = schedule[a]  # get the random operation based on the index
            del schedule[a]
            schedule.insert(0, element)

            # idx1, idx2 = random.sample(range(len(jobs_with_slack)), 2)
            # a, b = schedule.index(jobs_with_slack[idx1]), schedule.index(jobs_with_slack[idx2])
            # schedule[b], schedule[a] = schedule[a], schedule[b]
        # gc = self.gannt_chart(schedule)
        # self.plot_gannt(schedule)

        return schedule

    def longestmachinemutation(self, schedule):
        # gannt_chart = self.gannt_chart(schedule)
        # ctimes = []
        # for machine in range(self.problem.numberOfMachines):
        #     ctimes.append(gannt_chart[machine][-1][-1])
        # ltm = ctimes.index(max(ctimes))  # longest time machine
        ltm = random.sample(range(self.problem.numberOfMachines), 1)[0]
        ops = [i * self.problem.numberOfMachines + (ltm) for i in range(self.problem.numberOfMachines)]

        idx1, idx2 = random.sample(range(len(ops)), 2)
        a, b = schedule.index(ops[idx1]), schedule.index(ops[idx2])
        schedule[b], schedule[a] = schedule[a], schedule[b]
        return schedule

    # helper functions
    # def gannt_chart(self, schedule):
    #     """Compiles a scheduling on the machines given a permutation of jobs
    #     with no time gap checking"""
    #
    #     fForceOrder = True
    #
    #     # Note that using [[]] * m would be incorrect, as it would simply
    #     # copy the same list m times (as opposed to creating m distinct lists).
    #
    #     gantt_chart = [[] for _ in range(self.problem.numberOfMachines)]
    #
    #     for operation in schedule:
    #         machine_number = self.problem.operation_numbers_dictionary[operation][0]
    #         job_number = self.problem.operation_numbers_dictionary[operation][1]
    #         proc_time = self.problem.processing_times[operation]
    #
    #         # determine the processing times of the job on other machines
    #         time_interval_list = []
    #         for machine in range(self.problem.numberOfMachines):
    #             # dont check the machine to be scheduled since one job can be scheduled only once.
    #             # Check other machines if they have operations scheduled before
    #             if machine != machine_number and len(gantt_chart[machine]) != 0:
    #                 for j in range(len(gantt_chart[machine])):
    #                     # check the  job numbers on other machines
    #                     # and determine if the machine processed an operation of the job
    #                     # to be scheduled now
    #                     if gantt_chart[machine][j][0] == job_number:
    #                         # put completion times of the job on other machines into a list
    #                         s_time = gantt_chart[machine][j][1]  # start time of the job on the other machine
    #                         c_time = gantt_chart[machine][j][-1]  # completion time of the job on other machine
    #
    #                         time_interval_list.append((s_time, c_time))
    #                         time_interval_list.sort(key=lambda x: x[0])  # sort the list according to start time
    #
    #         # determine the completion time of the last operation (available time) on the required machine
    #         num_of_jobs_on_current_machine = len(gantt_chart[machine_number])
    #         if num_of_jobs_on_current_machine == 0:
    #             current_machine_available_time = 0
    #         else:  # buradan emin degilim
    #             current_machine_available_time = gantt_chart[machine_number][-1][-1]
    #
    #         if not fForceOrder:
    #             while True:
    #                 if len(time_interval_list) != 0:
    #                     for times in time_interval_list:
    #                         # intersection1 = range(max(current_machine_available_time, times[0]),
    #                         #                      min(current_machine_available_time + proc_time, times[1]))
    #                         intersection = min(current_machine_available_time + proc_time, times[1]) - max(
    #                             current_machine_available_time, times[0])
    #                         if intersection > 0:
    #                             current_machine_available_time = times[1]
    #                             f_intersection = True
    #                             break
    #                     else:
    #                         f_intersection = False
    #                 else:
    #                     break
    #
    #                 if not f_intersection:
    #                     break
    #             time_to_schedule = current_machine_available_time
    #         else:  # keep the order of schedule
    #             if len(time_interval_list) != 0:
    #                 ###########
    #                 end_time = current_machine_available_time + proc_time
    #                 ###########
    #
    #
    #                 for s_time, c_time in time_interval_list:
    #                     range_set = set(range(current_machine_available_time,
    #                                           current_machine_available_time + proc_time))
    #                     overlap = range_set.intersection(set(range(s_time, c_time)))
    #                     if overlap:
    #                         current_machine_available_time = c_time
    #
    #                         # previous_completion_times = [i[-1] for i in time_interval_list]
    #                         # max_prev_ctimes = max(previous_completion_times)
    #                         # time_to_schedule = max(max_prev_ctimes, current_machine_available_time)
    #                         # You made changes here
    #                     time_to_schedule = current_machine_available_time
    #             else:
    #                 time_to_schedule = current_machine_available_time
    #
    #         completion_time = time_to_schedule + proc_time
    #         gantt_chart[machine_number].append((job_number, time_to_schedule,
    #                                             proc_time, completion_time))
    #     return gantt_chart

    # OSSP_GA_OOP.py icinden aldigin class fonksiyonu
    def gannt_chart(self, schedule):
        """
        Compiles a scheduling on the machines given a permutation of jobs 
        with the option of time gap checking
        """

        # Note that using [[]] * m would be incorrect, as it would simply
        # copy the same list m times (as opposed to creating m distinct lists).

        gantt_chart = [[] for _ in range(self.problem.numberOfMachines)]

        for operation in schedule:
            machine_number = self.problem.operation_numbers_dictionary[operation][0]
            job_number = self.problem.operation_numbers_dictionary[operation][1]
            proc_time = self.problem.processing_times[operation]

            # determine the processing times of the job on other machines
            time_interval_list = []
            for machine in range(self.problem.numberOfMachines):
                # dont check the machine to be scheduled since one job can be scheduled only once.
                # Check other machines if they have operations scheduled before
                if machine != machine_number and len(gantt_chart[machine]) != 0:
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

            if not self.fForceOrder:  # eger kromozomdaki makinelere dusen is sirasi gozetilmeyecekse
                # first find the gaps on the machine to be scheduled
                if num_of_jobs_on_current_machine != 0:
                    gaps = []
                    current_starttime2check = 0
                    for op in (gantt_chart[machine_number]):
                        if current_starttime2check == op[1]:  # if start time equals to the operations start time
                            current_starttime2check = op[3]  # then update start time to operations end time
                        else:
                            gap = (current_starttime2check, op[1])
                            gaps.append(gap)
                            current_starttime2check = op[3]

                    # if there are gaps on the current machine check if it can be scheduled on those gaps
                    if gaps:
                        for space in gaps:
                            current_machine_available_time = space[0]
                            # Narrow the gap by checking other machines
                            time_to_schedule = self.check_overlap_othmach(time_interval_list, proc_time,
                                                                     current_machine_available_time)

                            space = (time_to_schedule, space[1])
                            # check if there is an overlap on the current machine
                            foverlap = self.check_overlap_curmach(space, proc_time)

                            if not foverlap:  # if there is no overlap on the current machine
                                time_to_schedule = space[0]
                                break
                            else:  # if operation doesnt fit in the gap
                                #  replace the available time with last operation's end time
                                current_machine_available_time = gantt_chart[machine_number][-1][-1]
                                time_to_schedule = self.check_overlap_othmach(time_interval_list, proc_time,
                                                                         current_machine_available_time)

                    else:  # if there are no gaps
                        current_machine_available_time = gantt_chart[machine_number][-1][-1]
                        time_to_schedule = self.check_overlap_othmach(time_interval_list, proc_time,
                                                                 current_machine_available_time)
                else:
                    current_machine_available_time = 0
                    time_to_schedule = self.check_overlap_othmach(time_interval_list, proc_time,
                                                             current_machine_available_time)
            else:  # keep the order of schedule
                if num_of_jobs_on_current_machine == 0:
                    current_machine_available_time = 0
                else:  # buradan emin degilim
                    current_machine_available_time = gantt_chart[machine_number][-1][-1]  # for order enforced case
                time_to_schedule = self.check_overlap_othmach(time_interval_list, proc_time, current_machine_available_time)

            completion_time = time_to_schedule + proc_time
            gantt_chart[machine_number].append((job_number, time_to_schedule,
                                                proc_time, completion_time))
            gantt_chart[machine_number].sort(key=lambda x: x[1])
        return gantt_chart

    @staticmethod
    def check_overlap_othmach(time_interval_list, proc_time, current_machine_available_time):
        if len(time_interval_list) != 0:
            for s_time, c_time in time_interval_list:
                range_set = set(range(current_machine_available_time,
                                      current_machine_available_time + proc_time + 1))
                overlap = range_set.intersection(set(range(s_time, c_time)))
                if overlap:
                    current_machine_available_time = c_time

            time_to_schedule = current_machine_available_time
        else:
            time_to_schedule = current_machine_available_time

        return time_to_schedule

    @staticmethod
    def check_overlap_curmach(gap, proc_time):
        if gap[1] - gap[0] < proc_time:
            return True
        return False

    def plot_gannt(self):
        """
        Plots the gannt chart of the given gannt chart data structure
        :return: None 
        """
        plt.ioff()
        fig, ax = plt.subplots()
        facecolors = ('blue', 'red', 'yellow', 'green', 'grey', 'azure', 'plum',
                      'wheat', 'brown', 'chocolate', 'coral', 'cyan', 'darkblue',
                      'gold', 'khaki', 'lavender', 'lime', 'magenta', 'orange',
                      'pink')
        bar_start = 10
        bar_width = 9
        increment = 10
        machine_times = self.gannt_chart(self.hof.items[0])
        # machine_times = self.gannt_chart(schedule)
        for i in range(self.problem.numberOfMachines):
            for j in range(self.problem.numberOfJobs):
                datalist = [machine_times[i][j][1:3]]
                ax.broken_barh(datalist, (bar_start, bar_width),
                               facecolors=facecolors[machine_times[i][j][0]])
            bar_start += increment

        ax.set_ylim(5, 115)
        ax.set_xlim(0, self.hof.items[0].fitness.values[0])
        ytickpos = range(15, 85, 10)
        ax.set_yticks(ytickpos)
        yticklabels = ['Machine ' + str(i + 1) for i in range(self.problem.numberOfMachines)]
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

    def evolve(self):
        for gen in range(1, self.NGEN):
            # Select the next generation individuals
            offspring = self.toolbox.select(self.pop, len(self.pop))

            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))  # clone offsprings

            if self.crossover == 'TCJCF':
                for child1, child2, child3 in zip(offspring[::3], offspring[1::3], offspring[2::3]):
                    if random.random() < self.CXPB:
                        self.toolbox.mate(child1, child2, child3)
                        del child1.fitness.values
                        del child2.fitness.values
                        # del child3.fitness.values
            else:
                # Apply crossover on the offspring
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < self.CXPB:
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

            # Apply mutation on the offspring
            for mutant in offspring:
                if random.random() < self.MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            if self.fApplySA:
                # apply simulated annealing to solutions
                for schedule in offspring:
                    sa = SA(schedule, self.problem)
                    sa.steps = 5
                    schedule, _ = sa.anneal()
                    del sa

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the individuals to survive based on the strategy
            if self.strategy == 'elitist1':
                self.pop[:] = tools.selBest(self.pop, 1) + \
                              tools.selBest(offspring, self.NPOP - 1)  # use elitism
            elif self.strategy == 'elitist2':
                self.pop[:] = tools.selBest(self.pop, 2) + \
                              tools.selBest(offspring, self.NPOP - 2)  # use elitism
            elif self.strategy == 'normal':
                self.pop[:] = offspring  # replace population with new offspring
            else:
                raise ValueError('Not a valid strategy')

            self.area_ratio = self.get_area_ratio()
            self.diversity = self.get_diversity()
            # Update the statistics of the new population
            self.update_stats(gen)

            # Check diversity of the population based on area under curve
            # if self.area_ratio < 0.5:
            #    self.apply_diversity()
            if self.fApplyDiversity:
                self.apply_diversity()

    def generate_population(self):
        # generate the initial population
        self.pop = self.toolbox.population(n=self.NPOP)
        # calculate the fitness values for individuals in the initial
        # population and assign them
        fitnesses = map(self.toolbox.evaluate, self.pop)
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit

    def init_stats(self):
        """
        This function initializes the statistic parameters
        that will be logged during the execution of the GA
        """

        self.hof = tools.HallOfFame(1)
        self.logbook = tools.Logbook()
        self.stats = tools.Statistics(key=lambda ind: ind.fitness.values)

        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

        # assign logbook headers
        self.logbook.header = "gen", "evals", "min", "max", "avg", "std", "ratio", "diversity"

        record = self.stats.compile(self.pop)
        self.logbook.record(gen=0, evals=len(self.pop), ratio=0.0, diversity=1, **record)
        if self.flag_print:
            print(self.logbook.stream)
        self.hof.update(self.pop)

    def update_stats(self, gen):
        """
        This function updates the statistic parameters
        that is logged during the execution of the GA
        """
        self.hof.update(self.pop)

        record = self.stats.compile(self.pop)
        self.logbook.record(gen=gen, evals=len(self.pop), ratio=self.area_ratio, diversity=self.diversity,
                            **record)
        if self.flag_print:
            print(self.logbook.stream)

    def register_functions(self):
        # define the problem as an Minimization or Maximization by defining the weights
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

        # create a container to hold individuals which have an attribute named "fitness"
        creator.create("Individual", array.array, typecode='B', fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()  # get toolbox class from base module of deap
        self.toolbox.register("attribute", random.sample, range(self.NDIM),
                              self.NDIM)  # register attribute method to toolbox
        self.toolbox.register("individual", tools.initIterate, creator.Individual,
                              self.toolbox.attribute)  # register individual method
        self.toolbox.register("population", tools.initRepeat, list,
                              self.toolbox.individual)  # register population method to toolbox

        if self.objective == 'makespan':
            # register objective function with the "evaluate" alias to the toolbox
            self.toolbox.register("evaluate", self.makespan)
        elif self.objective == 'total_tardiness':
            # register objective function with the "evaluate" alias to the toolbox
            self.toolbox.register("evaluate", self.total_tardiness)
        elif self.objective == 'total_completion_time':
            # register objective function with the "evaluate" alias to the toolbox
            self.toolbox.register("evaluate", self.sum_of_completion_times)
        else:
            raise ValueError('Not a valid objective function')

        # register crossover function with the "mate" alias to the toolbox
        if self.crossover == 'one_point':
            self.toolbox.register('mate', self.one_point_crossover)
        elif self.crossover == 'two_point':
            self.toolbox.register("mate", self.two_point_crossover)  # tools.cxOrdered
        elif self.crossover == 'ordered':
            self.toolbox.register('mate', tools.cxOrdered)
        elif self.crossover == 'linear_order':
            self.toolbox.register('mate', self.linear_order_crossover)
        elif self.crossover == 'TCJCF':
            self.toolbox.register('mate', self.three_chromosome_juggling_forward)
        elif self.crossover == 'gchart':
            self.toolbox.register('mate', self.gchart_crossover)
        else:
            raise ValueError('Not a valid crossover function')

        # register mutation function with the "mutate" alias to the toolbox
        if self.mutation == 'swap':
            self.toolbox.register("mutate", self.swap)
        elif self.mutation == 'shift':
            self.toolbox.register('mutate', self.shift)
        elif self.mutation == 'shuffle':
            self.toolbox.register('mutate', self.shuffle)
        elif self.mutation == 'inversion':
            self.toolbox.register('mutate', self.inversion)
        elif self.mutation == 'slacktime':
            self.toolbox.register('mutate', self.slacktimemutation)
        elif self.mutation == 'insertion':
            self.toolbox.register('mutate', self.insert)
        elif self.mutation == 'longesttime':
            self.toolbox.register('mutate', self.longestmachinemutation)
        else:
            raise ValueError('Not a valid mutation function')

        # register selection function with the "select" alias to the toolbox
        self.toolbox.register("select", tools.selTournament, tournsize=2)

    def print_best(self):
        print("Best Solution :", list(self.hof.items[0]))
        if self.objective == 'makespan':
            print('Best Makespan :', self.hof.items[0].fitness.values[0])
        elif self.objective == 'total_tardiness':
            print('Total Tardiness :', self.hof.items[0].fitness.values[0])

    def get_area_ratio(self):
        # get the generations array up to this point
        gen = self.logbook.select('gen')

        # get the minimum values of each generation
        min_val = self.logbook.select('min')

        if len(gen) >= 2:
            # calculate area under the curve of minimum values
            area = np.trapz([i - min(min_val) for i in min_val], gen)

            # calculate the triangular area from the starting point to ending point
            start_point = min_val[0]
            end_point = min_val[-1]
            total_area = (len(gen) - 1) * abs(start_point - end_point) / 2

            area_ratio = area / total_area
            if np.isnan(area_ratio) or np.isinf(area_ratio):
                area_ratio = 0.0
        else:
            area_ratio = 0.0

        return area_ratio

    def get_diversity(self):
        outer_sum = 0
        for i in range(self.NPOP):
            inner_sum = 0
            for j in range(self.NPOP):
                dist = self.hamming_distance(self.pop[i], self.pop[j])
                inner_sum += dist
            outer_sum += inner_sum
        diversity = (1 / (self.NPOP * self.NDIM * (self.NPOP - 1))) * outer_sum
        return diversity

    @staticmethod
    def hamming_distance(a, b):
        distance = 0
        if len(a) == len(b):
            for i in range(len(a)):
                if a[i] != b[i]:
                    distance += 1

        return distance

    def apply_diversity(self):  # mu
        if self.diversity_metric == 'distance':
            diversity = self.diversity
        elif self.diversity_metric == 'area':
            diversity = self.area_ratio
        else:
            raise ValueError('Not a valid metric')

        if diversity >= 0.5:
            self.MUTPB = 0.01
            # self.CXPB = 0.8
        elif 0.4 <= diversity < 0.5:
            self.MUTPB = 0.1
            # self.CXPB = 0.7
        elif 0.3 <= diversity < 0.4:
            self.MUTPB = 0.2
            # self.CXPB = 0.6
        elif 0.2 <= diversity < 0.3:
            self.MUTPB = 0.3
            # self.CXPB = 0.5
        elif 0.1 <= diversity < 0.2:
            self.MUTPB = 0.5
            # self.CXPB = 0.4
        elif diversity < 0.1:
            self.MUTPB = 0.6
            # self.CXPB = 0.2

    def plot_fitness(self):
        x = self.logbook.select('gen')
        y = self.logbook.select('min')
        plt.plot(x, y)
        plt.show()

    def find_operation_number(self, val):
        for key, value in self.problem.operation_numbers_dictionary.items():
            if value == val:
                return key

    def operation_scheduler(self, sequence):
        schedule = [[] for i in range(self.problem.numberOfMachines)]
        for operation in sequence:
            machine_no, job_no = self.problem.operation_numbers_dictionary[operation]
            schedule[machine_no].append(operation)
        return schedule

    def ltrpom(self, zipped_list):
        """
         Longest Total Remaining Processing on Other Machines
        :param zipped_list: 
        :return: 
        """


def main():
    # random.seed(8322)
    ossp_problem = Problem(filename='instances/Openshop/tai5_5.txt', instance=1)
    # print(OpenShopGA.hamming_distance(a, b))
    ossp_ga = OpenShopGA(ossp_problem, objective='makespan', mutation='swap', crossover='one_point',
                         max_gen=1000, pop_size=80, cross_pb=0.8, mut_pb=0.2, fprint=True,
                         strategy='elitist1', fApplyDiversity=False, diversity_metric='distance', fApplySA=False)
    ossp_ga.evolve()
    ossp_ga.print_best()
    ossp_ga.plot_gannt()
    ossp_ga.plot_fitness()


if __name__ == "__main__":
    main()
