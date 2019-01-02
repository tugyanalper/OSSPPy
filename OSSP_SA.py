from __future__ import print_function

import random
import sys

import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
import numpy as np
from simanneal import Annealer


class OpenShopSchedulingProblem(Annealer):
    """
    Test annealer with a open shop scheduling problem.
    """

    def __init__(self, filename, instance, objective):
        self.objective = objective
        self.processing_times, self.machines = self.parse_problem(filename, instance)
        self.numberOfJobs = len(self.processing_times)
        self.numberOfMachines = len(self.processing_times[0])
        self.dimension = self.numberOfJobs * self.numberOfMachines
        self.operation_numbers_dictionary = {i: (i % self.numberOfMachines, i // self.numberOfMachines)
                                             for i in range(self.dimension)}
        self.gannt_chrt = []
        self.due_dates = []
        state = random.sample(range(self.dimension), self.dimension)
        self.flatten_ptimes()
        super(OpenShopSchedulingProblem, self).__init__(state)  # important!

    def move(self):
        """Swaps two operations in the route."""
        a = random.randint(0, len(self.state) - 1)
        b = random.randint(0, len(self.state) - 1)
        self.state[a], self.state[b] = self.state[b], self.state[a]

    def movexx(self):
        """Swaps two operations in the route."""
        # state is the solution
        original_state = [[], [], [], [], []]
        E = []
        for i in range(5):
            original_state[i] = self.copy_state(self.state)
        for i in range(5):
            a = random.randint(0, len(original_state[i]) - 1)
            b = random.randint(0, len(original_state[i]) - 1)
            # print(original_state[i])

            original_state[i][a], original_state[i][b] = original_state[i][b], original_state[i][a]

            # print(original_state[i])
            gchart = self.gannt_chart(original_state[i])
            e = self.makespan(gchart)
            E.append(e)
        best_state_idx = E.index(min(E))
        self.state = original_state[best_state_idx]

    def energy(self):
        if self.objective == 'makespan':
            """Calculates the makespan of the schedule."""
            self.gannt_chrt = self.gannt_chart(self.state)
            ctimes = []
            for machine in range(self.numberOfMachines):
                ctimes.append(self.gannt_chrt[machine][-1][-1])
            make_span = max(ctimes)
            return make_span

        elif self.objective == 'total completion time':
            pass

        elif self.objective == 'total tardiness':
            self.gannt_chrt = self.gannt_chart()

            job_times = [[] for _ in range(self.numberOfJobs)]

            for machine in range(self.numberOfMachines):
                self.gannt_chrt[machine].sort(key=lambda x: x[0])
                for idx, job in enumerate(self.gannt_chrt[machine]):
                    job_times[idx].append((job[0], job[3]))

            ctimes = []
            for job in range(self.numberOfJobs):
                job_times[job].sort(key=lambda x: x[1])
                ctimes.append(job_times[job][-1][-1])

            tardiness = 0
            for idx, due_date in enumerate(self.due_dates):
                if ctimes[idx] > due_date:
                    tardiness += ctimes[idx] - due_date

            return tardiness

        else:
            raise ValueError('Not a valid value for objective function')

    @staticmethod
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

    def flatten_ptimes(self):
        new_ptimes = np.zeros((self.numberOfJobs, self.numberOfMachines), dtype=np.int16)
        for i, job in enumerate(self.processing_times):
            newlist = sorted(zip(self.machines[i], job))
            for j, ptime in enumerate(newlist):
                new_ptimes[i, j] = ptime[1]
        self.due_dates = np.sum(new_ptimes, axis=1)  # due dates are equal to total processing times of jobs
        self.processing_times = new_ptimes.flatten()

    def gannt_chart(self, state):
        """
        Compiles a scheduling on the machines given a permutation of jobs 
        with no time gap checking
        :return: gantt_chart list of list of tuples (job number, start time, processing time, completion time)
        """

        fForceOrder = True

        # Note that using [[]] * m would be incorrect, as it would simply
        # copy the same list m times (as opposed to creating m distinct lists).

        gantt_chart = [[] for _ in range(self.numberOfMachines)]

        for operation in state:
            machine_number = self.operation_numbers_dictionary[operation][0]
            job_number = self.operation_numbers_dictionary[operation][1]
            proc_time = self.processing_times[operation]

            # determine the processing times of the job on other machines

            time_interval_list = []
            for machine in range(self.numberOfMachines):
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

    def plot_gannt(self):
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
        ctimes = []
        for i in range(self.numberOfMachines):
            ctimes.append(self.gannt_chrt[i][-1][-1])
            for j in range(self.numberOfJobs):
                datalist = [self.gannt_chrt[i][j][1:3]]
                ax.broken_barh(datalist, (bar_start, bar_width),
                               facecolors=facecolors[self.gannt_chrt[i][j][0]])
            bar_start += increment
        xlimit = max(ctimes)
        ax.set_ylim(5, 115)
        ax.set_xlim(0, xlimit)
        ytickpos = range(15, 85, 10)
        ax.set_yticks(ytickpos)
        yticklabels = ['Machine ' + str(i + 1) for i in range(self.numberOfMachines)]
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

    def makespan(self, gchart):
        """Calculates the makespan of the schedule."""
        ctimes = []
        for machine in range(self.numberOfMachines):
            ctimes.append(gchart[machine][-1][-1])
        make_span = max(ctimes)
        return make_span


def main():
    filename = 'instances/Openshop/tai10_10.txt'
    instance_number = 1
    objective = 'makespan'
    ossp = OpenShopSchedulingProblem(filename, instance_number, objective)
    ossp.Tmax = 25000
    # auto_schedule =ossp.auto(minutes=1)
    # print(auto_schedule)
    # ossp.set_schedule(auto_schedule)
    ossp.steps = 20000
    # since our state is just a list, slice is the fastest way to copy
    ossp.copy_strategy = "slice"

    best_state, e = ossp.anneal()  # this is the main loop that executes simulated annealing

    print(best_state, e)
    ossp.state = best_state
    ossp.gannt_chrt = ossp.gannt_chart(ossp.state)
    for i in range(ossp.numberOfMachines):
        print(ossp.gannt_chrt[i])
    ossp.plot_gannt()


if __name__ == '__main__':
    main()
