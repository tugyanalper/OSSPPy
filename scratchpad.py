from __future__ import print_function, division

import sys

import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from itertools import groupby, chain


# # Read processing times from the file
# def parse_problem(filename, k=1):
#     """Parse the kth instance of a Taillard problem file
#
#     The Taillard problem files are a standard benchmark set for the problem
#     of flow shop scheduling. They can be found online at the following address:
#     - http://mistic.heig-vd.ch/taillard/problemes.dir/ordonnancement.dir/ordonnancement.html"""
#
#     with open(filename, 'r') as f:
#         # Identify the string that separates instances
#         problem_line = '/number of jobs, number of machines, time seed, machine seed, upper bound, lower bound :/'
#
#         # Strip spaces and newline characters from every line
#         lines = map(str.strip, f.readlines())
#
#         # We prep the first line for later
#         lines[0] = '/' + lines[0]
#
#         # We also know '/' does not appear in the files, so we can use it as
#         #  a separator to find the right lines for the kth problem instance
#         try:
#             proctimes = '/'.join(lines).split(problem_line)[k].split('/machines')[0].split('/')[2:]
#             machines = '/'.join(lines).split(problem_line)[k].split('/machines')[1].split('/')[1:]
#         except IndexError:
#             max_instances = len('/'.join(lines).split(problem_line)) - 1
#             print("\nError: Instance must be within 1 and %d\n" % max_instances)
#             sys.exit(0)
#
#         # Split every line based on spaces and convert each item to an int
#         data = [map(int, line.split()) for line in proctimes]
#
#         machines = [map(int, line.split()) for line in machines]
#
#     # We return the zipped data to rotate the rows and columns, making each
#     # item in data the durations of tasks for a particular job
#     return data, machines
#
#
# filename = 'instances/Openshop/tai5_5.txt'
# processing_times, machines = parse_problem(filename, 1)  # a list of [job number] [machine number]
# numberOfJobs = len(processing_times)
# numberOfMachines = len(processing_times[0])
# NDIM = numberOfMachines * numberOfJobs
# # print(processing_times)
#
#
# new_ptimes = []
# for idx, job in enumerate(processing_times):
#     newlist = sorted(zip(machines[idx], job))
#     ptimes_inorder = [element[1] for element in newlist]
#     new_ptimes.append(ptimes_inorder)
#
# numberOfMachines = 4
# numberOfJobs = 4
# processing_times = [item for sublist in new_ptimes for item in sublist]
# operation_numbers_dictionary = {i: (i % numberOfMachines, i // numberOfMachines)
#                                 for i in range(NDIM)}  # i : (machine number, job number)


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
        new_ptimes = np.zeros((self.numberOfJobs, self.numberOfMachines), dtype=np.uint8)
        for i, job in enumerate(self.processing_times):
            newlist = sorted(zip(self.machineOrder[i], job))
            for j, ptime in enumerate(newlist):
                new_ptimes[i, j] = ptime[1]
        self.due_dates = np.sum(new_ptimes, axis=1)  # tight due dates
        self.processing_times = new_ptimes.flatten()


def gannt_chart(problem, schedule):
    """ 
    Compiles a scheduling on the machines given a permutation of jobs 
    with no time gap checking
    """

    flag_print = False

    # Note that using [[]] * m would be incorrect, as it would simply
    # copy the same list m times (as opposed to creating m distinct lists).
    gantt_chart = [[] for _ in range(problem.numberOfMachines)]

    for operation in schedule:
        machine_number = problem.operation_numbers_dictionary[operation][0]
        job_number = problem.operation_numbers_dictionary[operation][1]
        proc_time = problem.processing_times[operation]

        # check if this job is being processed in any other machine
        completion_time_list = []
        time_interval_list = []
        for machine in range(problem.numberOfMachines):
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
                            time_interval_list.sort(key=lambda x: x[0])

        # determine the maximum completion time for this job on other machines
        if len(completion_time_list) != 0:
            other_machine_ending_time = max(completion_time_list)

        else:
            # this job has no previous operation
            other_machine_ending_time = 0

        # determine the completion time of the last operation (available time) on the required machine
        num_of_jobs_on_current_machine = len(gantt_chart[machine_number])
        if num_of_jobs_on_current_machine == 0:
            current_machine_available_time = 0
        else:
            current_machine_available_time = gantt_chart[machine_number][-1][-1]

        f_intersection = True
        while True:
            if len(time_interval_list) != 0:
                for times in time_interval_list:

                    intersection = range(max(current_machine_available_time, times[0]),
                                         min(current_machine_available_time + proc_time, times[1]))
                    # intersection = min(proc_time, times[1]) + 1 - max(current_machine_available_time, times[0])
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


def gannt_chart2(problem, schedule):
    """Compiles a scheduling on the machines given a permutation of jobs
    with checikng time gaps"""

    flag_print = False

    # Note that using [[]] * m would be incorrect, as it would simply
    # copy the same list m times (as opposed to creating m distinct lists).
    gantt_chart = [[] for _ in range(problem.numberOfMachines)]

    for operation in schedule:
        machine_number = problem.operation_numbers_dictionary[operation][0]
        job_number = problem.operation_numbers_dictionary[operation][1]
        proc_time = problem.processing_times[operation]

        # check if this job is being processed in any other machine
        completion_time_list = []
        time_interval_list = []
        for machine in range(problem.numberOfMachines):
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
                            time_interval_list.sort(key=lambda x: x[0])

        # determine the maximum completion time for this job on other machines
        if len(completion_time_list) != 0:
            other_machine_ending_time = max(completion_time_list)

        else:
            # this job has no previous operation
            other_machine_ending_time = 0

        # determine the completion time of the last operation (available time) on the required machine
        num_of_jobs_on_current_machine = len(gantt_chart[machine_number])
        gaps = []
        if num_of_jobs_on_current_machine == 0:
            current_machine_available_time = 0
            current_machine_available_time = check_intersection(time_interval_list, proc_time,
                                                                current_machine_available_time)
        else:
            # find gaps and put them in a list
            for i in range(num_of_jobs_on_current_machine):
                if i == 0:
                    first_job_start_time = gantt_chart[machine_number][i][1] - 0
                    if first_job_start_time > 0:
                        gaps.append((0, first_job_start_time))
                else:
                    following_job_start_time = gantt_chart[machine_number][i][1]
                    previous_job_end_time = gantt_chart[machine_number][i - 1][3]
                    time_gap = following_job_start_time - previous_job_end_time
                    if time_gap > 0:
                        gaps.append((previous_job_end_time, following_job_start_time))

            if len(gaps) == 0:
                # there are no gaps so the current available time is the last operation completion time
                current_machine_available_time = gantt_chart[machine_number][-1][-1]
                # check if there are no operation of the job on other machines and if there is an
                # intersection delay the current machine available time
                current_machine_available_time = check_intersection(time_interval_list, proc_time,
                                                                    current_machine_available_time)
            else:  # there are gaps
                flag_break_loop = False
                for times in gaps:
                    # check if the operation processing time fits in the time gaps on this machine
                    if times[1] - times[0] >= proc_time:
                        # If operation processing time fits the time gap, then current machine is available
                        # at the start of the time gap so set it with times[0]
                        current_machine_available_time = times[0]
                        if len(time_interval_list) != 0:
                            # check if this gap intersects with other machines
                            for times in time_interval_list:
                                intersection = min(current_machine_available_time + proc_time, times[1]) - max(
                                    current_machine_available_time, times[0])
                                if intersection > 0:
                                    break  # there is an intersection so break the loop
                            else:
                                # te job fits the time gap and it doesnt have intersection with other
                                # operations on other machines therefore schedule it in the time gap
                                # current_machine_available_time = times[0]
                                flag_break_loop = True
                        if flag_break_loop:
                            break
                else:
                    current_machine_available_time = gantt_chart[machine_number][-1][-1]
                    # check if there are no operation of the job on other machines and if there is an
                    # intersection delay the current machine available time
                    current_machine_available_time = check_intersection(time_interval_list, proc_time,
                                                                        current_machine_available_time)

        completion_time = current_machine_available_time + proc_time
        gantt_chart[machine_number].append((job_number, current_machine_available_time,
                                            proc_time, completion_time))
    return gantt_chart


# OSSP_GA_OOP.py icinden aldigin class fonksiyonu
def gannt_chart3(problem, schedule, flag):
    """
    Compiles a scheduling on the machines given a permutation of jobs 
    with the option of time gap checking
    """

    fForceOrder = flag
    fRepair = False

    # Note that using [[]] * m would be incorrect, as it would simply
    # copy the same list m times (as opposed to creating m distinct lists).

    gantt_chart = [[] for _ in range(problem.numberOfMachines)]

    for operation in schedule:
        machine_number = problem.operation_numbers_dictionary[operation][0]
        job_number = problem.operation_numbers_dictionary[operation][1]
        proc_time = problem.processing_times[operation]

        # determine the processing times of the job on other machines
        time_interval_list = []
        for machine in range(problem.numberOfMachines):
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

        if not fForceOrder:  # eger kromozomdaki makinelere dusen is sirasi gozetilmeyecekse
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
                        time_to_schedule = check_overlap_othmach(time_interval_list, proc_time,
                                                                 current_machine_available_time)

                        space = (time_to_schedule, space[1])
                        # check if there is an overlap on the current machine
                        foverlap = check_overlap_curmach(space, proc_time)

                        if not foverlap:  # if there is no overlap on the current machine
                            time_to_schedule = space[0]
                            fRepair = True
                            break
                        else:  # if operation doesnt fit in the gap
                            #  replace the available time with last operation's end time
                            current_machine_available_time = gantt_chart[machine_number][-1][-1]
                            time_to_schedule = check_overlap_othmach(time_interval_list, proc_time,
                                                                     current_machine_available_time)

                else:  # if there are no gaps
                    current_machine_available_time = gantt_chart[machine_number][-1][-1]
                    time_to_schedule = check_overlap_othmach(time_interval_list, proc_time,
                                                             current_machine_available_time)
            else:
                current_machine_available_time = 0
                time_to_schedule = check_overlap_othmach(time_interval_list, proc_time, current_machine_available_time)
        else:  # keep the order of schedule
            if num_of_jobs_on_current_machine == 0:
                current_machine_available_time = 0
            else:  # buradan emin degilim
                current_machine_available_time = gantt_chart[machine_number][-1][-1]  # for order enforced case
            time_to_schedule = check_overlap_othmach(time_interval_list, proc_time, current_machine_available_time)

        completion_time = time_to_schedule + proc_time
        gantt_chart[machine_number].append((job_number, time_to_schedule,
                                            proc_time, completion_time))
        gantt_chart[machine_number].sort(key=lambda x: x[1])
    if fRepair:
        repair_chromosome(schedule, gantt_chart, problem)
    return gantt_chart


def check_intersection(time_interval_list, proc_time, current_machine_available_time):
    while True:
        if len(time_interval_list) != 0:
            for times in time_interval_list:
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
    return current_machine_available_time


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


def check_overlap_curmach(gap, proc_time):
    if gap[1] - gap[0] < proc_time:
        return True
    return False


def repair_chromosome(schedule, gchart, problem):
    elements = problem.numberOfMachines * problem.numberOfJobs
    original = np.empty(elements, dtype=object)
    for i, operation in enumerate(schedule):
        original[i] = problem.operation_numbers_dictionary[operation]

    chart = deepcopy(gchart)
    dtype = [('operation', 'object'), ('start_time', 'int'), ('proc_time', 'int'), ('end_time', 'int')]
    chart = np.array(chart, dtype=dtype).flatten()

    for i, operation in enumerate(chart):
        machine_number = i // problem.numberOfMachines
        operation = ((machine_number, operation[0]), operation[1], operation[2], operation[3])
        chart[i] = operation

    chart = np.sort(chart, order='start_time')
    chart = [list(grp) for k, grp in groupby(chart, key=lambda x: x[1])]
    repaired = list(map(extract, chart))
    indices = {b: i for i, b in enumerate(original)}
    for sublist in repaired:
        if len(sublist) > 1:
            sublist.sort(key=lambda x: indices[x])
    del indices, chart, original, elements
    repaired = list(chain(*repaired))
    rev_dict = {value: key for key, value in problem.operation_numbers_dictionary.items()}
    repaired = list(map(lambda x: rev_dict[x], repaired))
    return repaired


def extract(sublist):
    if len(sublist) > 1:
        operations = [operation[0] for operation in sublist]
    else:
        operations = [sublist[0][0]]
    return operations


# Define Objective Function
def makespan(problem, schedule, flag):
    gannt_chrt = gannt_chart3(problem, schedule, flag)
    ctimes = []
    for machine in range(problem.numberOfMachines):
        ctimes.append(gannt_chrt[machine][-1][-1])
    make_span = max(ctimes)
    return gannt_chrt, make_span  # return a tuple for compatibility


def plot_gannt(machine_times, problem, ms):
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
    for i in range(problem.numberOfMachines):
        for j in range(problem.numberOfJobs):
            datalist = [machine_times[i][j][1:3]]
            ax.broken_barh(datalist, (bar_start, bar_width),
                           facecolors=facecolors[machine_times[i][j][0]])
        bar_start += increment

    ax.set_ylim(5, 115)
    ax.set_xlim(0, ms)
    ytickpos = range(15, 85, 10)
    ax.set_yticks(ytickpos)
    yticklabels = ['Machine ' + str(i + 1) for i in range(problem.numberOfMachines)]
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


def main():
    # random.seed(1250)
    ossp_problem = Problem(filename='instances/Openshop/tai4_4.txt', instance=1)
    schedule = np.array([12, 0, 7, 2, 6, 3, 13, 5, 1, 4, 14, 11, 10, 15, 8, 9])
    # schedule = np.array([12, 7, 2, 11, 6, 1, 0, 13, 9, 5, 14, 3, 10, 15, 4, 8]) #repaired
    fForceOrder = False
    print(schedule)
    gchart, ms = makespan(ossp_problem, schedule, fForceOrder)

    for idx, machine in enumerate(gchart):
        print('Machine ' + str(idx), ' :', machine)
    print(ms)
    plot_gannt(gchart, ossp_problem, ms)


if __name__ == '__main__':
    main()
