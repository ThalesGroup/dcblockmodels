"""Pairwise semi-supervised Latent Block Model with a Hidden Markov Random Field"""


import os
import sys
import warnings

import numpy as np
import scipy as sp
from numba import NumbaPendingDeprecationWarning

from .blockmodel import BlockModel
from .. import metrics
from .utils import (
    e_step,
    m_step,
    init,
    general,
    similarity_matrices,
    consensus
)

sys.stderr = open(os.devnull, "w")  # pylint: disable=R1732,W1514

warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class HLBM(BlockModel):
    """Semi-supervised Latent Block Model with a pairwise Hidden Markov Random Field

    Block Model General Parameters
    ----------
    Kz : int
        Number of row clusters
    Kw : int
        Number of column clusters
    init_type : {'random', 'kmeans', 'skmeans', 'given'}, optional,
        by default 'skmeans'. The methods used for the initialization of
        the algorithm. It is fit once for each side (rows and columns)
        and gives two global initialization partitions. These partitions
        are then used to initialization-specific partitions using the
        parameters `node_perturbation_rate` and `cluster_perturbation_rate`:
            - '`random'` randomly initialized clusters,
            - `'kmeans'`  clusters initialized using `sklearn.cluster.KMeans`,
            - `'skmeans'` clusters initialized using `spherecluster.SphericalKMeans`,
            - `'given'` given in the `fit()` method
    em_type : {'VEM', 'CEM'}
        The EM algorithm used
    n_init : int
        Number of initializations of the model. Each initialization
        will start from different initial row and col partitions
        and will converge to a partition and parameter set with
        a corresponding complete-data log likelihood. One can then
        select one of these partitions using the `best_partition()`
        method.
    n_init_clustering : int, optional
        the number of initializations of the clustering algorithm
        chosen in `'init_type'`, by default 100
    node_perturbation_rate : float, optional
        the fraction of nodes (row or cols) that are reassgined to
        a random cluster at each new intitialization of the model, by default .1
    model_type : {'with_margins', 'without_margins'}
        The dynamic model used:
            - `'with_margins'` : a model with dynamic margins
            and static connectivity matrix gamma
            - `'without_margins'` : a model with only a dynamic
            connectivity matrix as presented in Matias & Miele
    type_init_margins : {'ones', 'X.', 'random', 'given'}, optional
        How the margins are initialized, by default 'ones':
            - `'ones'` : mu[i] = nu[j] = 1
            - `'X.'` : mu = X.sum((1)) nu = X.sum((0))
            - `'random'` : mu and nu sampled from normal distribution
            followed by an absolute value.
            - `'given'` : mu an nu are given in the `fit()` method
    min_float : float, optional
        The minimum float used to avoid numerical issues, by default 1e-15
    min_gamma : float
        The minimum value of connectivity parameter gamma
        especially important in CEM to avoid empty clusters
    min_proba_Z : float, optional
        The probability (between 0 and 1) at which the variational probabilities
        for the row memberships are clipped, by default .05
    min_proba_W : float, optional
        The probability (between 0 and 1) at which the variational probabilities
        for the column memberships are clipped, by default .05
    min_margin : float, optional
        The value at which the margins are clipped, by default 1e-10
    min_proba_mixture_proportions : float, optional
        The probability (between 0 and 1) at which the mixture proportions
        are clipped, by default .05
    threshold_absent_nodes : int, optional
        Row or column nodes that have, at a given time step, a degree below
        `threshold_absent_nodes` are considered absent and do not contribute
        to the observed data log likelihood, by default 0
    dtype : str, optional
        The dtype of the floats used in the model, by default 'float32'
    random_state : int | np.random.RandomState | None, optional
        Creates a random state generator from a seed or uses the given
        random state generator, by default None
    max_iter : int, optional
        The maximum number of EM iterations for a single
        initialization of the model, by default 50
    tol_iter : float, optional
        The decrease ratio below which we consider the algorithm
        has converged, by default 1e-5
    n_jobs : int, optional
        The number of jobs used in the initialization clustering algorithm
        in sklearn or spherecluster, by default -1 i.e. all cores
    verbose : {0, 1, 2}, optional
        0 is silent, 1 is normal verbose and 2 is very verbose, by default 1
    blockmodel_params : [type], optional
        A dictionnary of parameters can can overwrite the class parameters
        if non empty or None, by default None. The parameters that can be changed
        are the following:
            - `n_iter_min`
            - `n_init_clustering_consensus`
            - `loc_random_margins`
            - `scale_random_margins`
    model_id : int | None, optional
        The numerical id of the model, used for debugging purposes, by default None
    debug_list : list, optional
        a list of strings that correspond to the names of the model attributes
        whose value we wish to keep track of during the iterations of the
        EM algorithm, by default []. `debug_list` must be a sublist of
        `self.model_parameters`.
    debug_output : str, optional
        The directory where the values of the parameters in `debug_list`
        will be outputed in the form of .npy files, by default '.'

    hLBM Specific Parameters
    ----------
    estimated_margins : True | False
        Whether the margins mu and nu are estimated or are
        set to the observed margins X.sum(1) and X.sum(0)
    regularization_mode : {'all', 'mixture'}
        Whether we consider the mixture proportions as
        an external field in the HMRF (`'all'`) or we consider
        the mixture proportions outside the HMRF (`'mixture'`)
    compute_regularization : bool, optional
        Whether we compute the regularization term in the criterion,
        which is computationally costly, by default True
    regularize_row : bool, optional
        Whether we use the similarity matrix S_r to regularize the
        model or not, by default False
    regularize_col : bool, optional
        Whether we use the similarity matrix S_c to regularize the
        model or not, by default False
    multiplicative_init_rows : bool, optional
        Whether we use the Must-Link relationships in the initialization
        of the row partition or not, by default True
    multiplicative_init_cols : bool, optional
        Whether we use the Must-Link relationships in the initialization
        of the column partition or not, by default True
    power_multiplicative_init : int if multiplicative_init_rows or
        if multiplicative_init_cols | None, optional
        The power to which we raise the stochastic matrix created from
        the Must-Link relationships in the initialization,
        by default None
    damping_factor : float if ((regularize_row or regularize_col) and
        em_type == 'VEM') | None, optional
        The damping factor between 0 and 1 used in VEM, by default None
    """
    REGULARIZATION_MODES = ['all', 'mixture']
    model_parameters = ['log_alpha', 'log_beta', 'gamma', 'mu', 'nu']

    def __init__(
            self,
            # Global blockmodel arguments
            Kz=None,
            Kw=None,
            init_type=None,
            em_type=None,
            n_init=None,
            n_init_clustering=100,
            node_perturbation_rate=.1,
            model_type=None,
            type_init_margins='ones',
            min_float=1e-15,
            min_gamma=None,
            min_proba_Z=.05,
            min_proba_W=.05,
            min_margin=1e-10,
            min_proba_mixture_proportions=.05,
            threshold_absent_nodes=0,
            dtype='float32',
            random_state=None,
            max_iter=50,
            tol_iter=1e-5,
            n_jobs=-1,
            verbose=1,
            blockmodel_params=None,
            model_id=None,
            debug_list=None,
            debug_output='.',
            # Specific HLBM arguments
            estimated_margins=None,
            regularization_mode=None,
            compute_regularization=True,
            regularize_row=False,
            regularize_col=False,
            multiplicative_init_rows=True,
            multiplicative_init_cols=True,
            power_multiplicative_init=None,
            damping_factor=None,
    ):
        super().__init__(
            Kz=Kz,
            Kw=Kw,
            init_type=init_type,
            em_type=em_type,
            n_init=n_init,
            min_float=min_float,
            n_init_clustering=n_init_clustering,
            node_perturbation_rate=node_perturbation_rate,
            model_type=model_type,
            type_init_margins=type_init_margins,
            min_gamma=min_gamma,
            min_proba_Z=.05,
            min_proba_W=.05,
            min_margin=1e-10,
            min_proba_mixture_proportions=min_proba_mixture_proportions,
            threshold_absent_nodes=threshold_absent_nodes,
            dtype=dtype,
            random_state=random_state,
            max_iter=max_iter,
            tol_iter=tol_iter,
            n_jobs=n_jobs,
            verbose=verbose,
            blockmodel_params=blockmodel_params,
            model_id=model_id,
            debug_list=debug_list,
            debug_output=debug_output
        )

        for arg in ['regularization_mode']:
            if arg is None:
                raise (f'Argument {arg} for class {type(self).__name__} must be initialized '
                       'explicitly. See documentation for possible values.')

        assert isinstance(multiplicative_init_rows, bool)
        assert isinstance(multiplicative_init_cols, bool)
        assert regularization_mode in self.REGULARIZATION_MODES
        assert 0. < min_proba_Z < 1.
        assert 0. < min_proba_W < 1.
        assert min_margin > 0.

        if self.model_type == 'with_margins':
            assert isinstance(estimated_margins, bool)
        else:
            assert not estimated_margins
        self.estimated_margins = bool(estimated_margins)

        self.regularization_mode = regularization_mode
        self.regularize_row = regularize_row
        self.regularize_col = regularize_col
        self.compute_regularization = compute_regularization

        if multiplicative_init_rows or multiplicative_init_cols:
            assert power_multiplicative_init >= 1
        self.multiplicative_init_rows = multiplicative_init_rows
        self.multiplicative_init_cols = multiplicative_init_cols
        self.power_multiplicative_init = power_multiplicative_init

        # damping_factor used for damping in HMRF VEM
        self.damping_factor = None
        if self.em_type == 'VEM' and (self.regularize_row or self.regularize_col):
            assert 0. < damping_factor < 1.
            self.damping_factor = damping_factor

        self.lambda_r, self.lambda_c = None, None
        self.S_r, self.S_c, self.P_r, self.P_c = None, None, None, None

    def fit(
        self,
        X,
        lambda_r=None,
        S_r=None,
        lambda_c=None,
        S_c=None,
        given_Z=None,
        given_W=None
    ):
        """Fits the model to the given data"""
        self.X = X
        self.N, self.D = self.X.shape
        self._set_similarity_parameters(lambda_r, S_r, lambda_c, S_c)
        self._set_global_init_partition(given_Z, given_W)
        self._init_debug()
        seeds = self.random_state.randint(np.iinfo(np.int32).max, size=self.n_init)

        (all_iter_criterions,
         all_intermediate_iter_criterions,
         all_regularizations,
         all_intermediate_regularizations,
         all_row_partitions,
         all_col_partitions) = [], [], [], [], [], []

        for i, _ in enumerate(seeds):
            self._print_verbose_msg_init(i)
            self._set_current_init_partition()
            self._init_q()
            self._init_parameters()

            old_iter_criterion = - np.finfo(np.float32).max
            new_iter_criterion = - np.finfo(np.float32).max
            iter_criterions, intermediate_iter_criterions = [], []
            regularizations, intermediate_regularizations = [], []

            for it in range(self.max_iter):
                self._print_verbose_msg_iter(it)
                self._debug(i)

                # E + M steps
                _old_interm_criterion, old_iter_criterion = new_iter_criterion, new_iter_criterion
                interm_criterion, interm_reg, new_iter_criterion, new_reg = self._fit_single()

                if self.regularize_row or self.regularize_col:
                    interm_criterion -= interm_reg
                    new_iter_criterion -= new_reg

                self._set_best_parameters(new_iter_criterion, i, it)
                delta_iter = general.get_delta(old_iter_criterion, new_iter_criterion)

                if (it >= self.n_iter_min) and (delta_iter < self.tol_iter):
                    self._print_verbose_converged(it, None)
                    break

                iter_criterions.append(new_iter_criterion)
                intermediate_iter_criterions.append(interm_criterion)
                regularizations.append(new_reg)
                intermediate_regularizations.append(interm_reg)

            all_row_partitions.append(general.to_dense(self.Z.copy()).argmax(1))
            all_col_partitions.append(general.to_dense(self.W.copy()).argmax(1))
            all_iter_criterions.append(iter_criterions)
            all_intermediate_iter_criterions.append(intermediate_iter_criterions)
            all_regularizations.append(regularizations)
            all_intermediate_regularizations.append(intermediate_regularizations)

        self.all_row_partitions = all_row_partitions
        self.all_col_partitions = all_col_partitions
        self.all_iter_criterions = all_iter_criterions
        self.all_intermediate_iter_criterions = all_intermediate_iter_criterions
        self.all_regularizations = all_regularizations
        self.all_intermediate_regularizations = all_intermediate_regularizations
        self.fitted = True
        self._set_mixture_proportions()
        self._write_best_parameters()

        return self

    def _m_step(self, X_red, mode):
        # update mixture proportions (alpha, beta)
        Z_ = general.get_class_counts(self.Z)
        W_ = general.get_class_counts(self.W)

        # alpha, beta
        self.update_mixture_proportions(mode, Z_, W_)

        # gamma
        X_kl, den_gamma = m_step.update_gamma(
            mode, self.estimated_margins,
            self.X, self.Z, self.W, X_red, self.mu, self.nu,
            self.dtype, self.model_type
        )
        self.gamma = m_step.get_gamma(
            X_kl, den_gamma,
            self.em_type, self.min_float, self.min_gamma
        )

        # mu, nu
        if self.estimated_margins:
            self.mu = m_step.update_mu(
                self.Z, self.W, self.Xi_, self.nu, self.gamma, self.min_margin
            )
            self.nu = m_step.update_nu(
                self.Z, self.W, self.X_j, self.mu, self.gamma, self.min_margin
            )

        # complete data log likelihood
        Lc = m_step.compute_Lc_static(
            self.model_type, self.estimated_margins,
            self.regularization_mode, self.regularize_row, self.regularize_col,
            self.P_r, self.P_c,
            self.log_alpha, self.log_beta, self.gamma,
            self.mu, self.nu, self.Xi_, self.X_j,
            Z_, W_, X_kl, self.min_float
        )

        # regulariation term of the complete data log likelihood
        # can be not computed for faster computations
        if self.compute_regularization:
            regularization = m_step.get_regularization(
                self.regularize_row, self.regularize_col,
                self.lambda_r, self.lambda_c,
                self.S_r, self.S_c, self.Z, self.W
            )
        else:
            regularization = 0.
        return Lc, regularization

    def _e_step(self, mode):
        if mode == 'row':
            self.Z, X_W = e_step.e_step_static(
                X=self.X, gamma=self.gamma,
                mode='row',
                model_type=self.model_type, em_type=self.em_type,
                estimated_margins=self.estimated_margins,
                regularization_mode=self.regularization_mode,
                regularize=self.regularize_row,
                lambda_=self.lambda_r, S=self.S_r, P=self.P_r,
                Z=self.Z, W=self.W,
                log_alpha_beta=self.log_alpha,
                mu=self.mu, nu=self.nu,
                damping_factor=self.damping_factor,
                dtype=self.dtype,
                min_proba=self.min_proba_Z, min_float=self.min_float
            )
            return X_W
        if mode == 'col':
            self.W, X_Z = e_step.e_step_static(
                X=self.X, gamma=self.gamma,
                mode=mode,
                model_type=self.model_type, em_type=self.em_type,
                estimated_margins=self.estimated_margins,
                regularization_mode=self.regularization_mode,
                regularize=self.regularize_col,
                lambda_=self.lambda_c, S=self.S_c, P=self.P_c,
                Z=self.Z, W=self.W,
                log_alpha_beta=self.log_beta,
                mu=self.mu, nu=self.nu,
                damping_factor=self.damping_factor,
                dtype=self.dtype,
                min_proba=self.min_proba_Z, min_float=self.min_float
            )
            return X_Z
        raise ValueError(self, "mode", mode)

    def _fit_single(self):
        """
        A single iteration of EM
        X_Z and X_W are the reduced matrices
        the M step returns the complete data log likelihood
        and the regularization term
        """
        X_W = self._e_step(mode='row')
        L1, reg1 = self._m_step(X_W, 'row')

        X_Z = self._e_step(mode='col')
        L2, reg2 = self._m_step(X_Z, 'col')
        return L1, reg1, L2, reg2

    def _init_parameters(self):
        # log_alpha, beta, gamma initialized in the first M-step
        self.gamma = np.zeros((self.Kz, self.Kw), dtype=self.dtype)
        self.log_alpha = np.zeros((self.Kz), dtype=self.dtype)
        self.log_beta = np.zeros((self.Kw), dtype=self.dtype)

        if sp.sparse.issparse(self.X):
            self.Xi_ = self.X.sum(1).A1
            self.X_j = self.X.sum(0).A1
        else:
            self.Xi_ = self.X.sum(1)
            self.X_j = self.X.sum(0)

        if self.estimated_margins:
            self.mu = np.ones((self.N), dtype=self.dtype)
            self.nu = np.ones((self.D), dtype=self.dtype)
        else:
            self.mu, self.nu = self.Xi_, self.X_j
            np.clip(self.mu, self.min_margin, None, self.mu)
            np.clip(self.nu, self.min_margin, None, self.nu)

        # first M step
        self._m_step(None, mode='init')

    def _set_global_init_partition(self, given_Z, given_W):
        # X_init : data matrix used for initial clustering of the rows
        X_init = init.get_X_init(
            self.X,
            mode='row',
            absent_nodes=None
        )
        if self.regularize_row and self.multiplicative_init_rows:
            X_init = similarity_matrices.init_transform(
                X_init, self.S_r, self.power_multiplicative_init
            )
        self.global_init_Z = init.get_init_partition(
            X_init=X_init,
            Kzw=self.Kz,
            init_type=self.init_type,
            T=1,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            n_init=self.n_init_clustering,
            given_partition=given_Z
        )
        # X_init : data matrix used for initial clustering of the cols
        X_init = init.get_X_init(
            self.X,
            mode='col',
            absent_nodes=None
        )
        if self.regularize_col and self.multiplicative_init_cols:
            X_init = similarity_matrices.init_transform(
                X_init, self.S_c, self.power_multiplicative_init
            )
        self.global_init_W = init.get_init_partition(
            X_init=X_init,
            Kzw=self.Kw,
            init_type=self.init_type,
            T=1,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            n_init=self.n_init_clustering,
            given_partition=given_W
        )

    def _set_current_init_partition(self):
        self.current_init_Z = init.apply_perturbation_to_init_clustering(
            init_partition=self.global_init_Z,
            random_state=self.random_state,
            node_perturbation_rate=self.node_perturbation_rate,
            dynamic=False
        )
        self.current_init_W = init.apply_perturbation_to_init_clustering(
            init_partition=self.global_init_W,
            random_state=self.random_state,
            node_perturbation_rate=self.node_perturbation_rate,
            dynamic=False
        )

    def _init_q(self):
        if self.em_type == 'VEM':
            self.Z = init.init_ZW0(
                Kzw=self.Kz,
                partition=self.current_init_Z,
                min_proba=self.min_proba_Z,
                dtype=self.dtype
            )
            self.W = init.init_ZW0(
                Kzw=self.Kw,
                partition=self.current_init_W,
                min_proba=self.min_proba_W,
                dtype=self.dtype
            )
        elif self.em_type == 'CEM':
            self.Z = init.init_ZW0(
                Kzw=self.Kz,
                partition=self.current_init_Z,
                min_proba=0,
                dtype='bool'
            )
            self.Z = general.to_sparse(self.Z)

            self.W = init.init_ZW0(
                Kzw=self.Kw,
                partition=self.current_init_W,
                min_proba=0,
                dtype='bool'
            )
            self.W = general.to_sparse(self.W)

    def update_mixture_proportions(self, mode, Z_, W_):
        """Update mixture proportions (alpha, beta) for rows and columns"""
        if mode in ['row', 'all']:
            self.log_alpha = Z_ / self.N
            np.clip(self.log_alpha, self.min_proba_mixture_proportions, None, self.log_alpha)
            np.divide(self.log_alpha, self.log_alpha.sum(), self.log_alpha)
            self.log_alpha = np.log(self.log_alpha + self.min_float)
        if mode in ['col', 'all']:
            self.log_beta = W_ / self.D
            np.clip(self.log_beta, self.min_proba_mixture_proportions, None, self.log_beta)
            np.divide(self.log_beta, self.log_beta.sum(), self.log_beta)
            self.log_beta = np.log(self.log_beta + self.min_float)

    def icl(self):
        """
        Returns the ICL values for each init of the model,
        sorted from best to worse complete data log likelihood
        """
        assert hasattr(self, 'best_parameters')
        penality = (
            0.5 * (self.Kz - 1) * np.log(self.N) +
            0.5 * (self.Kw - 1) * np.log(self.D) +
            0.5 * self.Kz * self.Kw * np.log(self.N * self.D)
        )
        icls = [params[0] - penality for params in self.best_parameters]
        return icls

    def _set_similarity_parameters(self, lambda_r, S_r, lambda_c, S_c):
        """
        P_r and P_c : ndarray of length N and D such that
        P_r[i] = True if we have prior knowledge about node i
        that is there exists i' such that S_r[i, i'] > 0
        """
        if self.regularize_row:
            assert lambda_r > 0
            similarity_matrices.check_similarity_matrix(S_r)

            self.lambda_r = lambda_r
            self.S_r = general.to_sparse(S_r)

            # since S is symmetric
            self.P_r = np.asarray((self.S_r != 0.).sum(0))[0] > 0
        else:
            self.lambda_r = None
            self.S_r = None
            self.P_r = np.zeros(self.N, dtype='bool')

        if self.regularize_col:
            assert lambda_c > 0
            similarity_matrices.check_similarity_matrix(S_c)

            self.lambda_c = lambda_c
            self.S_c = general.to_sparse(S_c)
            self.P_c = np.asarray((self.S_c != 0.).sum(0))[0] > 0
        else:
            self.lambda_c = None
            self.S_c = None
            self.P_c = np.zeros(self.D, dtype='bool')

    def _set_mixture_proportions(self):
        self.alpha = np.exp(self.log_alpha)
        self.beta = np.exp(self.log_beta)

    def best_partition(self, mode, n_first=1):
        """
        returns a list of tuple of partitions
            if 1 partition: [(part_1), ..., (part_n_first)]
            if 2 partitions: [(row_part_1, col_part_1), ...,
                                (row_part_n_first, col_part_n_first)]

        mode == 'likeelihood' returns the n_first best partitions
        in terms of likelihood

        if mode == 'consensus: hbgf' or mode == 'consensus: cspa'
         returns 1 consensus partition
        """
        assert self.fitted

        best_partitions = metrics.sort_partitions(
            self.all_iter_criterions,
            [self.all_row_partitions, self.all_col_partitions],
            n_first
        )
        if mode == 'likelihood':
            return best_partitions

        if mode in ['consensus: hbgf', 'consensus: cspa']:
            best_row_partitions = [x[0] for x in best_partitions]
            best_col_partitions = [x[1] for x in best_partitions]

            if mode == 'consensus: hbgf':
                Z_consensus = consensus.hbgf(
                    best_row_partitions,
                    self.n_init_clustering_consensus
                )
                W_consensus = consensus.hbgf(
                    best_col_partitions,
                    self.n_init_clustering_consensus
                )
            elif mode == 'consensus: cspa':
                Z_consensus = consensus.cspa(
                    best_row_partitions,
                    self.n_init_clustering_consensus
                )
                W_consensus = consensus.cspa(
                    best_col_partitions,
                    self.n_init_clustering_consensus
                )
            return [(Z_consensus, W_consensus)]

        raise ValueError
