"""Base class for the Block Models"""

import time
import os
import sys
import warnings
import pickle

import numpy as np
import scipy as sp
from numba import NumbaPendingDeprecationWarning
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

np.seterr(all='raise')

# FIXME Probably a way to remove warnings? (problem in the whole program)
sys.stderr = open(os.devnull, "w")  # pylint: disable=R1732,W1514

warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class BlockModel(BaseEstimator):
    """Base class for the dynamic and semi-supervised LBM"""

    model_parameters = []  # filled in subclasses

    # The default blockmodel parameters
    # overwritten by the params given in dic blockmodel_params
    # in the construction of BlockModel
    n_iter_min = 5
    n_init_clustering_consensus = 100
    loc_random_margins = 1e-8  # mean and std used for the initialization
    scale_random_margins = 1e-3  # of the margins if self.type_init_margins == 'random'

    # checking the inputs
    MODEL_TYPES = [
        'with_margins',  # static & dynamic
        'without_margins'  # static & dynamic
    ]
    EM_TYPES = ['CEM', 'VEM']
    INITS_TYPES = [
        'random',
        'given',
        'kmeans',
        'skmeans'
    ]
    TYPES_INIT_MARGINS = ['ones', 'X.', 'random', 'given']
    DTYPES = ['float32', 'float64']
    VERBOSE_LEVELS = [0, 1, 2]

    def __init__(
            self,
            Kz,
            Kw,
            init_type,
            em_type,
            n_init,
            n_init_clustering,
            node_perturbation_rate,
            model_type,
            type_init_margins,
            min_float,
            min_gamma,
            min_proba_Z,
            min_proba_W,
            min_margin,
            min_proba_mixture_proportions,
            threshold_absent_nodes,
            dtype,
            random_state,
            max_iter,
            tol_iter,
            n_jobs,
            verbose,
            blockmodel_params,
            model_id,
            debug_list,
            debug_output
    ):
        super().__init__()

        for arg in ['Kz', 'Kw', 'init_type', 'em_type', 'n_init', 'model_type', 'min_gamma']:
            if arg is None:
                raise (f'Argument {arg} for class {type(self).__name__} must be initialized '
                       'explicitly. See documentation for possible values.')

        assert Kz > 1
        assert Kw > 1

        assert init_type in self.INITS_TYPES
        assert em_type in self.EM_TYPES
        assert model_type in self.MODEL_TYPES

        assert dtype in self.DTYPES
        assert verbose in self.VERBOSE_LEVELS
        assert type_init_margins in self.TYPES_INIT_MARGINS

        assert min_float > 0.
        assert min_gamma >= 0.
        assert 0. < min_proba_mixture_proportions < 1.

        self.Kz = Kz
        self.Kw = Kw

        self.init_type = init_type
        self.em_type = em_type
        self.n_init = n_init
        self.n_init_clustering = n_init_clustering
        self.node_perturbation_rate = node_perturbation_rate
        self.type_init_margins = type_init_margins

        self.min_float = min_float
        self.min_gamma = min_gamma
        self.min_proba_Z = min_proba_Z
        self.min_proba_W = min_proba_W
        self.min_margin = min_margin
        self.min_proba_mixture_proportions = min_proba_mixture_proportions
        self.threshold_absent_nodes = threshold_absent_nodes

        self.dtype = dtype
        self.random_state = check_random_state(random_state)

        self.max_iter = max_iter
        self.tol_iter = tol_iter

        self.n_jobs = n_jobs
        self.verbose = verbose

        self.blockmodel_params = blockmodel_params
        self._set_blockmodel_params(blockmodel_params)

        self.model_id = int(time.time()) if model_id is None else model_id
        self.debug_list = debug_list if debug_list is not None else []
        self.debug_output = debug_output

        self.cluster_perturbation_rate = None

        self.N, self.D, self.T = None, None, None
        self.X = None
        self.W, self.Z = None, None
        self.alpha, self.beta = None, None
        self.gamma, self.mu, self.nu = None, None, None
        self.log_alpha, self.log_beta, self.log_pi, self.log_rho = None, None, None, None
        self.Xi_, self.X_j = None, None

        self.qz, self.qw = None, None
        self.current_init_W, self.current_init_Z = None, None
        self.global_init_W, self.global_init_Z = None, None

        self.best_parameters, self.best_criterion = None, None

        self.model_type = model_type

        self.debug_counts = None
        self.debug_path = None

        self.fitted = False

        self.all_row_partitions = None
        self.all_col_partitions = None
        self.all_iter_criterions = None
        self.all_intermediate_iter_criterions = None
        self.all_icls = None
        self.all_regularizations = None
        self.all_intermediate_regularizations = None

    def _set_blockmodel_params(self, blockmodel_params):
        """
        overwrites the default params with given params in __init__
        """
        params_names = [
            "min_proba_mixture_proportions",
            "min_float_margins",
            "n_iter_min",
            "loc_random_margins",
            "scale_random_margins",
        ]
        if blockmodel_params is not None:
            for param_name, param_value in blockmodel_params.items():
                if param_name in params_names:
                    setattr(self, param_name, param_value)

    # ################### Verbose #################### #

    def _print_verbose_msg_iter(self, n_iter):
        if self.verbose == 1:
            if (n_iter % 5 == 4) or (n_iter == self.max_iter - 1):
                print(f'     Iteration {n_iter + 1} on {self.max_iter}')
        elif self.verbose == 2:
            print(f'     Iteration {n_iter + 1} on {self.max_iter}')

    def _print_verbose_msg_init(self, n_init):
        if self.verbose >= 1:
            print()
            print(f'*** Model {self.model_id}: '
                  f'initialization {n_init + 1} on {self.n_init} ***')

    def _print_verbose_converged(self, n_iter_tot, ind_smoothing):
        if self.verbose >= 1:
            if ind_smoothing is None:
                print(f'     {n_iter_tot + 1} Iterations  are enough')
            else:
                if ind_smoothing % 5 == 4:
                    print(f'     smoothing step {ind_smoothing + 1} at iter {n_iter_tot + 1}')

    def _print_verbose_smoothing(self, smoothing_schedule, n_iter_supp_smoothing):
        if self.verbose >= 1:
            n_steps = len(smoothing_schedule)
            n_iter_supp = n_steps * n_iter_supp_smoothing
            print(
                f'     Smoothing: {n_iter_supp} new iter max ({n_steps} steps, '
                f'{n_iter_supp_smoothing} iter max per tau)'
            )

    # ################### Debug #################### #

    def _init_debug(self):
        """
        creates appropriate directories where
        the parameters we want to debug will be saved
        """
        self.debug_output.mkdir(exist_ok=True)

        # FIXME Verify that the path is a Pathlib.path!

        # FIXME And create directory dynamically!
        def _unique_dir(base_directory, name_pattern):
            c = 0
            while True:
                c += 1
                directory = base_directory / name_pattern.format(c)
                if not directory.is_dir():
                    return directory

        # for each init, the number of times debug() was called
        self.debug_counts = np.zeros((self.n_init), dtype='int')

        # name_pattern = 'debug_model_{}'.format(self.model_id) + '_{:03d}'
        # TODO verify behaviour
        name_pattern = f'debug_model_{self.model_id}_{{:03d}}'
        self.debug_path = _unique_dir(self.debug_output, name_pattern)

        if len(self.debug_list) > 0:
            os.mkdir(self.debug_path)
            for debug_item in self.debug_list:
                os.mkdir(self.debug_path / debug_item)

    def _debug(self, init_nb):
        """
        e.g. debug_list = ['Z', 'gamma']
            adds Z_0.npy in dir /Z and gamma_0.npy in /gamma
            then adds Z_1.npy and gamma_1.npy
            etc ...

        call the method every time you want to log
        the state of a given variable
        """
        for debug_item in self.debug_list:
            item = getattr(self, debug_item, None)
            item_id = f'{debug_item}_init_{init_nb}_{self.debug_counts[init_nb]}'
            item_path = self.debug_path / debug_item / item_id
            _save_item(item, item_path)

        self.debug_counts[init_nb] += 1

    def has_sparse_ZW(self):
        """Check whether self.Z is either sparse or a list"""
        return sp.sparse.issparse(self.Z) or isinstance(self.Z, list)

    def get_debug(self, sublist=None):
        """
        returns a dictionnary containing the values of each
        item in self.debug_list (e.g. debug_list = ['Z', 'gamma'])
        at each iteration of the algorithm
        if sublist is not None, only returns the elements that are
        both in self.debug_list and sublist
        """
        if sublist is not None:
            debug_list = [x for x in sublist if x in self.debug_list]
        else:
            debug_list = self.debug_list

        res = {}
        for debug_item in debug_list:
            if debug_item in ['Z', 'W']:
                ext = '.npz' if self.has_sparse_ZW() else '.npy'
            else:
                ext = '.npy'
            res_item = []  # one list for a given item, e.g. gamma
            for init_nb in range(self.n_init):
                res_init = []  # one list for each initialization of an item
                for debug_count in range(self.debug_counts[init_nb]):
                    item_id = (debug_item + f'_init_{init_nb}_{debug_count}' + ext)
                    item_path = self.debug_path / debug_item / item_id
                    item = _load_item(item_path)
                    res_init.append(item)
                res_item.append(res_init)
            res[debug_item] = res_item
        return res

    # ################### Model utils #################### #

    def _set_best_parameters(self, criterion, cur_init, cur_iter):
        if cur_init == 0 and cur_iter == 0:
            self.best_parameters = [[] for _ in range(self.n_init)]

        if cur_iter == 0:
            self.best_criterion = criterion
            self._set_new_best_params(cur_init)
        else:
            if criterion > self.best_criterion:
                self.best_criterion = criterion
                self._set_new_best_params(cur_init)

    def _set_new_best_params(self, cur_init):
        best_params = [self.best_criterion]
        for param_name in self.model_parameters:
            best_params.append(getattr(self, param_name))
        self.best_parameters[cur_init] = best_params

    def _write_best_parameters(self):
        """
        writes attribute best_parameters
        [(criterion(init_i),
          {'param_name': best_value_param_init_i, ...}), ...]
        ordered by criterion
        """
        self.best_parameters = [
            (self.best_parameters[i][0],
             dict(zip(self.model_parameters, self.best_parameters[i][1:])))
            for i in range(self.n_init)
        ]
        self.best_parameters = sorted(self.best_parameters, key=lambda x: x[0], reverse=True)

    def save(self, path='.', modelname=None):
        """
        Saves the model at path path with the name modelname as a pickle file.
        """
        if modelname is None:
            modelname = f'{type(self).__name__}_saved_at_{int(time.time())}'

        full_path = path + '/' + modelname
        os.makedirs(full_path, exist_ok=True)

        model_path = full_path + '/' + 'model.pickle'
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)
        print('Model saved at : ' + model_path)


def _save_item(item, path):
    if isinstance(item, np.ndarray):
        np.save(path.with_suffix('.npy'), item)
    elif sp.sparse.issparse(item):
        sp.sparse.save_npz(path.with_suffix('.npz'), item)
    else:
        raise TypeError


def _load_item(path):
    if path.suffix == '.npy':
        return np.load(path, allow_pickle=True)
    if path.suffix == '.npz':
        return sp.sparse.load_npz(path)
    raise TypeError
