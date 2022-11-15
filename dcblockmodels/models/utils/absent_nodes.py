"""class to efficiently deal with the indexes of absent and appearing nodes"""

import functools

import numpy as np
from numba import njit


class AbsentNodes:
    """
    class of nodes that dont interract with others at a given
    time step, i.e. their in and out degrees are below/above threshold

    both : True for dSBM, False for dLBM
    """

    def __init__(self, X, threshold, mode):

        self.T = len(X)

        if mode == 'SBM':
            (n_absent_tot, n_appearing_tot,
             absent, appearing, n_abs, n_app) = present_and_absent_nodes(X, threshold, 'both')

            (self.n_absent_row_tot,
             self.n_appearing_row_tot,
             self.absent_row_nodes,
             self.appearing_row_nodes,
             self.n_absent_row_nodes,
             self.n_appearing_row_nodes) = (n_absent_tot, n_appearing_tot,
                                            absent, appearing, n_abs, n_app)

            (self.n_absent_col_tot,
             self.n_appearing_col_tot,
             self.absent_col_nodes,
             self.appearing_col_nodes,
             self.n_absent_col_nodes,
             self.n_appearing_col_nodes) = (n_absent_tot, n_appearing_tot,
                                            absent, appearing, n_abs, n_app)

        elif mode == 'LBM':
            (self.n_absent_row_tot,
             self.n_appearing_row_tot,
             self.absent_row_nodes,
             self.appearing_row_nodes,
             self.n_absent_row_nodes,
             self.n_appearing_row_nodes) = present_and_absent_nodes(X, threshold, 'row')

            (self.n_absent_col_tot,
             self.n_appearing_col_tot,
             self.absent_col_nodes,
             self.appearing_col_nodes,
             self.n_absent_col_nodes,
             self.n_appearing_col_nodes) = present_and_absent_nodes(X, threshold, 'col')

        self.set_replace_vals_absent()

    def __str__(self):
        return ('Row nodes : \n'
                f'n_absent_row_nodes = {self.n_absent_row_nodes}\n'
                f'absent_row_nodes = {self.absent_row_nodes}\n'
                f'n_appearing_row_nodes = {self.n_appearing_row_nodes}\n'
                f'appearing_row_nodes = {self.appearing_row_nodes}\n\n'
                'Column nodes : \n'
                f'n_absent_col_nodes = {self.n_absent_col_nodes}\n'
                f'absent_col_nodes = {self.absent_col_nodes}\n'
                f'n_appearing_col_nodes = {self.n_appearing_col_nodes}\n'
                f'appearing_col_nodes = {self.appearing_col_nodes}\n')

    def __repr__(self):
        return self.__str__()

    def set_replace_vals_absent(self):
        """
        previous timestep at which
        node i was present
        """
        self.inds_prev_rows = {}
        self.inds_prev_cols = {}
        self.ts_absent_rows = {}
        self.ts_absent_cols = {}

        for absent_dic, inds_prev_dic, timesteps_absent in zip(
            [self.absent_row_nodes, self.absent_col_nodes],
            [self.inds_prev_rows, self.inds_prev_cols],
            [self.ts_absent_rows, self.ts_absent_cols]
        ):
            inds_absent = functools.reduce(
                np.union1d, (absent_dic[t] for t in range(self.T))
            ).astype('int64')

            for i in inds_absent:
                ts_absent = [t for t in range(self.T) if i in absent_dic[t]]
                inds_prev_dic[i] = inds_prev_absent(self.T, ts_absent)
                timesteps_absent[i] = np.array(ts_absent)


def present_and_absent_nodes(X, threshold, axis):
    """
    Assess nodes that are absent at each time, deduce present & appearing nodes at each time

    Absent nodes are nodes that dont interact with others at a given time step, i.e. their in and
    out degrees are below/above threshold.
    """

    if isinstance(X, np.ndarray):
        T = X.shape[0]
        if axis == 'row':
            nodes_degrees = X.sum(2)
        elif axis == 'col':
            nodes_degrees = X.sum(1)
        elif axis == 'both':
            nodes_out_degrees = X.sum(1)
            nodes_in_degrees = X.sum(2)
            nodes_degrees = nodes_in_degrees + nodes_out_degrees
    elif isinstance(X, list):
        T = len(X)
        if axis == 'row':
            nodes_degrees = np.array([X[t].sum(1).A1 for t in range(T)])
        elif axis == 'col':
            nodes_degrees = np.array([X[t].sum(0).A1 for t in range(T)])
        elif axis == 'both':
            nodes_out_degrees = np.array([X[t].sum(0).A1 for t in range(T)])
            nodes_in_degrees = np.array([X[t].sum(1).A1 for t in range(T)])
            nodes_degrees = nodes_in_degrees + nodes_out_degrees

    absent = np.where(nodes_degrees <= threshold)

    absent_dic = {}
    appearing_dic = {0: np.array([], dtype='int')}

    n_absent_tot, n_appearing_tot = 0, 0
    n_absent_dic, n_appearing_dic = {}, {}
    for t in range(T):
        ind0_t = np.where(absent[0] == t)
        absent_dic[t] = absent[1][ind0_t]
        n_absent_tot += len(absent_dic[t])
        if t >= 1:
            appearing_t = []
            for i in absent_dic[t - 1]:
                if i not in absent_dic[t]:
                    appearing_t.append(i)
                    n_appearing_tot += 1
            appearing_dic[t] = np.array(appearing_t, dtype='int')

        n_absent_dic[t] = len(absent_dic[t])
        n_appearing_dic[t] = len(appearing_dic[t])

    return (n_absent_tot, n_appearing_tot, absent_dic,
            appearing_dic, n_absent_dic, n_appearing_dic)


def replace_vals_absent(arr, inds_prev, ts_absent):
    """
    Replace values for an absent node by previous values at which it was present
    """
    for i, ts_absent_i, inds_prev_i in zip(
        ts_absent.keys(),
        ts_absent.values(),
        inds_prev.values()
    ):
        arr[ts_absent_i, i] = arr[inds_prev_i, i]


@njit
def inds_prev_absent(T, inds):
    """Computes the timesteps used to replace the values
    of the smoothed arrays for absent nodes

    Parameters
    ----------
    T : int
        number timesteps
    inds : list
        indexes of the timesteps at which a a given node is absent

    Returns
    -------
    np.ndarray
        array of same size as inds that maps each timestep
        at which a a given node is absent to the previous timestep
        at which the node was present
    """
    inds_prev = []
    i_prev = -1
    len_inds = 0
    for i in range(T):
        if i not in inds:
            i_next = i
            break
    for i in range(T):
        if i in inds:
            len_inds += 1
            if i_prev != -1:
                inds_prev.append(i_prev)
            else:
                inds_prev.append(i_next)
        else:
            i_prev = i
        if len_inds == T:
            break
    return np.array(inds_prev, dtype=np.int64)
