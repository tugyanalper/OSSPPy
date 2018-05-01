from __future__ import print_function, division

import random
import sys

import matplotlib.pyplot as plt


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
            lines = '/'.join(lines).split(problem_line)[k].split('/machines')[0].split('/')[2:]
        except IndexError:
            max_instances = len('/'.join(lines).split(problem_line)) - 1
            print("\nError: Instance must be within 1 and %d\n" % max_instances)
            sys.exit(0)

        # Split every line based on spaces and convert each item to an int
        data = [map(int, line.split()) for line in lines]

    # We return the zipped data to rotate the rows and columns, making each
    # item in data the durations of tasks for a particular job
    return zip(*data)


filename = 'instances/Openshop/tai5_5.txt'
processing_times = parse_problem(filename, 1)  # a list of [job number] [machine number]
numberOfJobs = len(processing_times)
numberOfMachines = len(processing_times[0])
NDIM = numberOfMachines * numberOfJobs
# print(processing_times)

operation_numbers_dictionary = {i: (i % numberOfMachines, i // numberOfMachines)
                                for i in range(NDIM)}


# Define Objective Function
def makespan(schedule):
    """Compiles a scheduling on the machines given a permutation of jobs"""

    flag_print = False

    # Note that using [[]] * m would be incorrect, as it would simply
    # copy the same list m times (as opposed to creating m distinct lists).
    gantt_chart = [[] for _ in range(numberOfMachines)]

    for operation in schedule:
        fInProcess = False
        machine_number = operation_numbers_dictionary[operation][0]
        job_number = operation_numbers_dictionary[operation][1]
        proc_time = processing_times[job_number][machine_number]

        # check if this job is being processed in any other machine
        completion_time_list = []
        for machine in range(numberOfMachines):
            if machine != machine_number:
                if len(gantt_chart[machine]) != 0:
                    for j in range(len(gantt_chart[machine])):
                        if gantt_chart[machine][j][0] == job_number:
                            # put completion times of job on other machines into a list
                            completion_time_list.append(gantt_chart[machine][j][-1])
                            fInProcess = True

        # determine the maximum completion time for this job on other machines
        if len(completion_time_list) != 0:
            other_machine_ending_time = max(completion_time_list)
        else:
            # this job has no previous operation
            other_machine_ending_time = 0

        try:
            lastjobidx = len(gantt_chart[machine_number]) - 1
            current_machine_available_time = gantt_chart[machine_number][lastjobidx][-1]
        except IndexError:
            current_machine_available_time = 0

        # last job completion time on current machine
        if fInProcess:
            start_time = max(other_machine_ending_time,
                             current_machine_available_time)
        else:
            start_time = current_machine_available_time
        completion_time = start_time + proc_time
        gantt_chart[machine_number].append((job_number, start_time,
                                            proc_time, completion_time))

        # schedule the current operation
        # if len(gantt_chart[machine_number]) == 0:
        #     completion_time = start_time + proc_time
        #     gantt_chart[machine_number].append((job_number, start_time,
        #                                         proc_time, completion_time))
        # else:
        #     lastjobidx = len(gantt_chart[machine_number]) - 1
        #     # last job completion time
        #     current_machine_available_time = gantt_chart[machine_number][lastjobidx][-1]
        #     if fInProcess:
        #         start_time = max(other_machine_ending_time,
        #                          current_machine_available_time)
        #     else:
        #         start_time = current_machine_available_time
        #     completion_time = start_time + proc_time
        #     gantt_chart[machine_number].append((job_number, start_time,
        #                                         proc_time, completion_time))

        if flag_print:
            print('--' * 20)
            print('Operation Number is : {}'.format(operation))
            print('Machine Number is : {}'.format(machine_number))
            print('Job Number is : {}'.format(job_number))
            print("Processing time for job {} on machine {} : ".format(job_number, machine_number), proc_time)
    ctimes = []
    for machine in range(numberOfMachines):
        ctimes.append(gantt_chart[machine][-1][-1])
    make_span = max(ctimes)
    return gantt_chart, make_span  # return a tuple for compatibility


def plot_gannt(machine_times, make_span):
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

    ax.set_ylim(5, 65)
    ax.set_xlim(0, make_span)
    ax.set_yticks([15, 25, 35, 45, 55])
    yticklabels = ['Machine ' + str(i + 1) for i in range(numberOfMachines)]
    ax.set_yticklabels(yticklabels)
    ax.grid(True)
    plt.show()


def main():
    # random.seed(1250)
    schedule = random.sample(range(NDIM), NDIM)
    # print(schedule)
    machine_times, make_span = makespan(schedule)
    # print(len(machine_times[0]))
    for machine in range(numberOfMachines):
        print("Gant Chart for Machine {}".format(machine + 1))
        print(machine_times[machine])
    print(make_span)
    plot_gannt(machine_times, make_span)

if __name__ == '__main__':
    main()
