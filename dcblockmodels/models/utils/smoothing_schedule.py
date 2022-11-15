"""Smoothing schedule for dLBM"""

import numpy as np
import matplotlib.pyplot as plt


class SmoothingSchedule:
    """Smoothing schedule for dLBM"""

    def __init__(self, schedule_type, length, tau0=1e-3, x0=-6., x1=6.):
        self.schedule_type = schedule_type
        self.length = length

        self.x0 = x0
        self.x1 = x1
        self.tau0 = tau0

        if self.schedule_type == 'sigmoid':
            schedule = np.linspace(x0, x1, self.length)
            # TODO check
            # func = lambda x: 1. / (1. + np.exp(- x))
            # schedule = func(schedule)
            schedule = 1. / (1. + np.exp(- schedule))

        elif self.schedule_type == 'linear':
            schedule = np.linspace(tau0, 1., self.length)

        # TODO check whether this is desirable (keeping a last schedule value != 1 amounts to a sort
        # of regularization)
        schedule[-1] = 1.

        self.schedule = schedule

    def plot(self):
        """Plot the schedule in Matplotlib"""
        f, ax = plt.subplots()
        ax.plot(self.schedule)
        f.suptitle('Smoothing Schedule')
        return ax

    def __str__(self):
        s = (
            'Smoothing schedule:\n'
            f'schedule type : {self.schedule_type}\n'
            f'{self.length} steps\n'
        )
        if self.schedule_type == 'linear':
            s += f'from {self.tau0} to 1.'
        else:
            s += f'sigmoid on [{self.x0}, {self.x1}]'
        return s

    def __repr__(self):
        return self.__str__()
