import os
import sys
import warnings

import pathlib
import pytest

import numpy as np
from sklearn.metrics import adjusted_rand_score

from ..models.hlbm import HLBM
from .. import data
from ..models.utils import general, similarity_matrices


sys.stderr = open(os.devnull, "w")

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Model test thresholds
ARI_ROW = {
    'easy': .4,
    'medium': .3,
    'hard': .1
}
ARI_COL = ARI_ROW.copy()


# Model fixed params
debug_output = pathlib.Path(r'./dcblockmodels/model_debug_output')
random_state = None
n_jobs = -1
verbose = 0
debug_list = []
power_multiplicative_init = 1  # True, False
n_iter_supp_smoothing = 5

frac_r, frac_c = .01, .01
frac_noise = 0.
regularization_mode = 'all'
lambda_0 = 2.
lambda_r, lambda_c = lambda_0, lambda_0
S_r, S_c = None, None
damping_factor = .7

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
n_init_clustering = 20
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
dtype = 'float64'
self_loops = True


class TestHLBM:
    # Data variable params
    # T, N, D, Kz, Kw, level, gamma_0, with_margins
    test_data_setups = [
        (100, 120, 3, 4, 'easy', True),  # incresing difficulty
        (100, 120, 3, 4, 'medium', True),  # incresing difficulty
        (100, 120, 3, 4, 'hard', True),  # incresing difficulty
        (100, 100, 3, 3, 'medium', True),  # N = D
        (500, 520, 3, 4, 'medium', True)  # N and D big
    ]

    n_init = 20
    n_first = 10  # n_first partitions that will be checked in the test, must be <= n_int

    def fit_model(
            self,
            test_data,
            sparse_X,
            estimated_margins,
            regularize,
            em_type,
            multiplicative_init
    ):
        (X, Z, W, N, D, Kz, Kw,
         level, with_margins) = test_data

        print('Test data = \n')
        print('level', level)
        print('with_margins', with_margins)

        model_type = 'with_margins' if with_margins else 'without_margins'
        regularize_row, regularize_col = regularize, regularize
        multiplicative_init_rows, multiplicative_init_cols = (
            multiplicative_init,
            multiplicative_init
        )
        S_r = general.to_sparse(
            similarity_matrices.build_S(Z, frac_r, frac_noise).astype(dtype)
        )
        S_c = general.to_sparse(
            similarity_matrices.build_S(W, frac_c, frac_noise).astype(dtype)
        )
        if sparse_X:
            X_ = general.to_sparse(X)
        else:
            X_ = X.copy()

        model = HLBM(
            Kz=Kz, Kw=Kw,
            model_type=model_type,
            estimated_margins=estimated_margins,
            regularization_mode=regularization_mode,
            regularize_row=regularize_row, regularize_col=regularize_col,
            n_init=TestHLBM.n_init,
            max_iter=max_iter,
            em_type=em_type,
            damping_factor=damping_factor,
            multiplicative_init_rows=multiplicative_init_rows,
            multiplicative_init_cols=multiplicative_init_cols,
            power_multiplicative_init=power_multiplicative_init,
            min_float=min_float,
            min_proba_Z=min_proba_Z,
            min_proba_W=min_proba_W,
            min_proba_mixture_proportions=min_proba_mixture_proportions,
            min_margin=min_margin,
            min_gamma=min_gamma,
            init_type=init_type,
            n_init_clustering=n_init_clustering,
            node_perturbation_rate=node_perturbation_rate,
            model_id=1,
            dtype='float64',
            threshold_absent_nodes=threshold_absent_nodes,
            blockmodel_params=blockmodel_params,
            random_state=None,  # np.random.RandomState(42)
            tol_iter=tol_iter,
            n_jobs=-1,
            verbose=0, debug_list=[],
            debug_output=debug_output
        )
        model.fit(
            X_,
            given_Z=given_Z, given_W=given_W,
            S_r=S_r, lambda_r=lambda_r, S_c=S_c, lambda_c=lambda_c
        )
        return model, Z, W, level

    def assert_metrics(self, model, Z, W, level, n_first):
        for init, (Z_model, W_model) in enumerate(model.best_partition(mode='likelihood',
                                                                       n_first=n_first)):
            ari_row_f = adjusted_rand_score(Z.flatten(), Z_model.flatten())
            ari_col_f = adjusted_rand_score(W.flatten(), W_model.flatten())

            print(f'level = {level}, ari global = {(ari_row_f, ari_col_f)}')

            assert ari_row_f > ARI_ROW[level]
            assert ari_col_f > ARI_COL[level]

    @pytest.fixture
    def test_data(self, request):
        (N, D, Kz, Kw,
         level, with_margins) = request.param

        model_type = 'LBM'
        dimensions = {'N': N, 'D': D}
        n_clusters = {'Kz': Kz, 'Kw': Kw}
        T = 1
        directed = True
        dtype = 'int32'

        alphas_dirichlet = {
            'very_easy': 10,
            'easy': 8,
            'medium': 6,
            'hard': 4
        }
        gamma0_level_dic = {
            'easy': 0.02,
            'medium': 0.01,
            'hard': 0.005
        }
        alpha = data.generate_initial_proportions(Kz, alphas_dirichlet[level])
        beta = data.generate_initial_proportions(Kw, alphas_dirichlet[level])
        prior_init = {'alpha': alpha, 'beta': beta}

        prior_trans = {'pi': None, 'rho': None}

        gamma_0 = gamma0_level_dic[level]

        # Data fixed params

        block_sparsity_matrix = None  # \beta_{kl}^t of Matias
        # block_sparsity_matrix = 0.1 * np.ones((Kz, Kw), dtype='float')

        constant_margins = True  # True, False
        start, stop, step = 1, 50, .1
        order_power_law = -1.5  # margins ~ Unif(start, stop)^order_power_law

        if with_margins:
            mu, nu = data.generate_margins(
                T, N, D, constant_margins, start, stop, step,
                directed, order_power_law,
                ar_margins=None, a_ar=None, sigma2_ar=None
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

        with_absent_nodes = False
        absent_nodes = {
            'absent_row_nodes': [],
            'absent_col_nodes': []
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
        return (X, Z, W, N, D, Kz, Kw,
                level, with_margins)
