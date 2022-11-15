import os
import sys
import warnings

import pathlib
import pytest

import numpy as np
from sklearn.metrics import adjusted_rand_score

from ..models.dlbm import dLBM
from .. import data
from ..models.utils import general
from ..models.utils.smoothing_schedule import SmoothingSchedule


sys.stderr = open(os.devnull, "w")

warnings.filterwarnings("ignore", category=DeprecationWarning)


# Model test thresholds
ARI_ROW_F = {
    'easy': .4,
    'medium': .15,
    'hard': .1
}
ARI_COL_F = ARI_ROW_F.copy()
ARI_ROW_T = ARI_ROW_F.copy()
ARI_COL_T = ARI_ROW_T.copy()

# Model fixed params
smoothing_schedule = SmoothingSchedule('sigmoid', 20)

debug_output = pathlib.Path(r'./dcblockmodels/model_debug_output')
random_state = None
n_jobs = -1
verbose = 0
debug_list = []
n_iter_supp_smoothing = 5

diag_pi_init = 0.7
diag_rho_init = 0.7
prior_diagonal_pi = 0.
prior_diagonal_rho = 0.

max_iter = 100
tol_iter = 1e-6
min_float = 1e-15
min_proba_Z, min_proba_W = .05, .05
min_proba_mixture_proportions = 1e-2  # to avoid empty clusters
min_margin = 1e-10
min_gamma = 1e-8

init_type = 'skmeans'  # 'skmeans', 'kmeans'
given_mu, given_nu = None, None
given_Z, given_W = None, None
n_init_clustering = 10
node_perturbation_rate = .15
cluster_perturbation_rate = 0.
threshold_absent_nodes = 0
blockmodel_params = {
    "n_iter_min": 5,
    "loc_random_margins": 1e-8,  # mean and std used for the initialization
    "scale_random_margins": 1e-3,  # of the margins if self.type_init_margins == 'random'
    "n_init_clustering_consensus": 100  # for best_partition() method of model
}
parameter_smoothing = True
type_init_margins = 'ones'


class TestDLBM:

    # Data variable params
    # T, N, D, Kz, Kw, level, gamma_0, with_margins, with_absent_nodes
    test_data_setups = [
        (10, 100, 120, 3, 4, 'easy', True, False),  # incresing difficulty
        (10, 100, 120, 3, 4, 'medium', True, False),  # incresing difficulty
        (10, 100, 120, 3, 4, 'hard', True, False),  # incresing difficulty
        (10, 100, 100, 3, 3, 'medium', True, False),  # N = D, Kz = Kw
        (10, 100, 120, 3, 4, 'medium', True, True),  # Absent nodes
        (100, 100, 120, 3, 4, 'medium', True, False),  # T big
        (10, 500, 800, 3, 4, 'medium', True, False)  # N and D big
    ]

    n_init = 10
    n_first = 2  # n_first partitions that will be checked in the test, must be <= n_int

    def fit_model(
            self,
            test_data,
            sparse_X,
            em_type,
            dtype='float64'):

        (X, Z, W, T, N, D, Kz, Kw,
         level, with_margins, with_absent_nodes) = test_data

        print('Test data = \n')
        print('level', level)
        print('with_margins', with_margins)
        print('with_absent_nodes', with_absent_nodes)

        model_type = 'with_margins' if with_margins else 'without_margins'

        if sparse_X:
            X_ = [general.to_sparse(X[t]) for t in range(T)]
        else:
            X_ = X.copy()

        model = dLBM(
            model_type=model_type,
            em_type=em_type,
            parameter_smoothing=parameter_smoothing,
            Kz=Kz, Kw=Kw,
            n_init=TestDLBM.n_init,
            model_id=1,
            max_iter=max_iter,
            type_init_margins=type_init_margins,
            smoothing_schedule=smoothing_schedule.schedule,
            n_iter_supp_smoothing=n_iter_supp_smoothing,
            prior_diagonal_pi=prior_diagonal_pi,
            prior_diagonal_rho=prior_diagonal_rho,
            diag_pi_init=diag_pi_init,
            diag_rho_init=diag_rho_init,
            init_type=init_type,
            n_init_clustering=n_init_clustering,
            node_perturbation_rate=node_perturbation_rate,
            cluster_perturbation_rate=cluster_perturbation_rate,
            threshold_absent_nodes=threshold_absent_nodes,
            min_proba_mixture_proportions=min_proba_mixture_proportions,
            min_gamma=min_gamma,
            min_margin=min_margin,
            min_proba_Z=min_proba_Z,
            min_proba_W=min_proba_W,
            dtype=dtype,
            blockmodel_params=blockmodel_params,
            random_state=random_state,
            tol_iter=tol_iter,
            n_jobs=n_jobs, verbose=verbose,
            debug_list=debug_list,
            debug_output=debug_output
        )
        model.fit(
            X_,
            given_Z=given_Z, given_W=given_W,
            given_mu=given_mu, given_nu=given_nu
        )
        return model, Z, W, level

    def assert_metrics(self, model, Z, W, level, n_first):
        for init, (Z_model, W_model) in enumerate(model.best_partition(mode='likelihood',
                                                                       n_first=n_first)):
            ari_row_f = adjusted_rand_score(Z.flatten(), Z_model.flatten())
            ari_col_f = adjusted_rand_score(W.flatten(), W_model.flatten())

            print(f'level = {level}, ari global = {(ari_row_f, ari_col_f)}')

            assert ari_row_f > ARI_ROW_F[level]
            assert ari_col_f > ARI_COL_F[level]

            for t in range(model.T):
                ari_row_t = adjusted_rand_score(Z[t], Z_model[t])
                ari_col_t = adjusted_rand_score(W[t], W_model[t])
                assert ari_row_t > ARI_ROW_T[level]
                assert ari_col_t > ARI_COL_T[level]

    @pytest.fixture
    def test_data(self, request):
        (T, N, D, Kz, Kw,
         level, with_margins, with_absent_nodes) = request.param

        model_type = 'LBM'
        dimensions = {'N': N, 'D': D}
        n_clusters = {'Kz': Kz, 'Kw': Kw}

        alphas_dirichlet = {
            'very_easy': 10,
            'easy': 8,
            'medium': 6,
            'hard': 4
        }
        diag_vals = {
            'diag': 0,
            'easy': .9,
            'medium': .75,
            'hard': .6
        }

        gamma0_level_dic = {
            'easy': 0.02,
            'medium': 0.01,
            'hard': 0.005
        }

        alpha = data.generate_initial_proportions(Kz, alphas_dirichlet[level])
        beta = data.generate_initial_proportions(Kw, alphas_dirichlet[level])
        prior_init = {'alpha': alpha, 'beta': beta}

        pi = data.generate_diag_transition_matrix(Kz, diag_vals[level])
        rho = data.generate_diag_transition_matrix(Kw, diag_vals[level])
        prior_trans = {'pi': pi, 'rho': rho}

        gamma_0 = gamma0_level_dic[level]

        # Data fixed params

        block_sparsity_matrix = None  # \beta_{kl}^t of Matias
        # block_sparsity_matrix = 0.1 * np.ones((Kz, Kw), dtype='float')

        constant_margins = True  # True, False
        start, stop, step = 1, 50, .1
        order_power_law = -1.5  # margins ~ Unif(start, stop)^order_power_law
        # mu ~ AR1 : mu_{t+1} = N(a mu_t + c, sigma2)
        # (c s.t. mu increasing if sigma2 = 0)
        ar_margins, a_ar, sigma2_ar = False, 1.1, .1

        # absent nodes
        min_proba_t = .0
        max_proba_t = .2
        proba_absent = None

        directed = True
        self_loops = True
        dtype = 'int32'

        if with_margins:
            mu, nu = data.generate_margins(
                T, N, D, constant_margins, start, stop, step,
                directed, order_power_law,
                ar_margins, a_ar, sigma2_ar
            )
            margins = {'mu': mu, 'nu': nu}
        else:
            margins = None

        noise_level_ = 0.

        if Kz == 3 and Kw == 4:
            gamma = gamma_0 * np.array([
                [1, 2, 4, 1],
                [3, 1, 2, 3],
                [2, 3, 1, 3]
            ])
        elif Kz == 3 and Kw == 3:
            gamma = gamma_0 * np.array([
                [1, 2, 3],
                [3, 1, 2],
                [2, 3, 1]
            ])
        else:
            raise ValueError

        if T > 1:
            gamma = np.stack([gamma for _ in range(T)], axis=0)
            if block_sparsity_matrix is not None:
                block_sparsity_matrix = np.stack([block_sparsity_matrix for _ in range(T)], axis=0)

        if with_absent_nodes:
            absent_row_nodes = data.sample_absent_nodes(
                T, N,
                min_proba_t=min_proba_t,
                max_proba_t=max_proba_t,
                proba_absent=proba_absent
            )
            if not directed:
                absent_col_nodes = absent_row_nodes.copy()
            else:
                absent_col_nodes = data.sample_absent_nodes(
                    T, D,
                    min_proba_t=min_proba_t,
                    max_proba_t=max_proba_t,
                    proba_absent=proba_absent
                )
        else:
            absent_row_nodes, absent_col_nodes = [], []

        absent_nodes = {
            'absent_row_nodes': absent_row_nodes,
            'absent_col_nodes': absent_col_nodes
        }

        X, Z, W = data.generate_data(
            T,
            model_type,
            dimensions,
            n_clusters,
            prior_init,
            prior_trans,
            gamma,
            with_margins,
            margins,
            self_loops,
            directed,
            noise_level_,
            with_absent_nodes,
            absent_nodes,
            dtype,
            block_sparsity_matrix
        )

        return (X, Z, W, T, N, D, Kz, Kw,
                level, with_margins, with_absent_nodes)
