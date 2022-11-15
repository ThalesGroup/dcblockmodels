"""Dynamic Latent Block Model"""

import os
import sys

import warnings
import numpy as np
from numba import NumbaPendingDeprecationWarning

from dcblockmodels import metrics
from .blockmodel import BlockModel
from .utils import (
    init,
    e_step,
    m_step,
    general,
    consensus,
    absent_nodes
)

sys.stderr = open(os.devnull, "w")  # pylint: disable=R1732,W1514

warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class dLBM(BlockModel):
    """Dynamic Latent Block Model

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
            - `'ones'` : mu[t, i] = nu[t_, j] = 1
            - `'X.'` : mu = X.sum((2)) nu = X.sum((1))
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
        This value can be set this to -1 to avoid considering 0-degree nodes as absent
        # TODO Verify compatibility with the rest of the code
    dtype : str, optional
        The dtype of the floats used in the model, by default 'float32'
    random_state : int | np.random.RandomState | None, optional
        Creates a random state generator from a seed or uses the given
        random state generator, by default None
    max_iter : int, optional
        The maximum number of EM iterations for a single
        initialization of the model, by default 500
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

    dLBM Specific Parameters
    ----------
    n_iter_supp_smoothing : int, optional
        The maximum number of smoothing iterations for a given step of
        the smoothing schedule, by default 5
    parameter_smoothing : bool
        Whether we apply parameter smoothing or not
    smoothing_schedule : class SmoothingSchedule
        Describes how the parameters will be smoothed during the
        iterations of the EM algorithm
    diag_pi_init : float, optional
        The value of each entry of the diagonal of the transition
        matrix pi at initialization, by default None which results
        in pi being first estimated using a M-step
    diag_rho_init : float, optional
        The value of each entry of the diagonal of the transition
        matrix pi at initialization, by default None which results
        in rho being first estimated using a M-step
    prior_diagonal_pi : float, optional
        Used as an informative prior for the diagonal terms of pi,
        and produces pseudocounts for intr-cluster transitions,
        by default 0.
    prior_diagonal_rho : float, optional
        Used as an informative prior for the diagonal terms of rho,
        and produces pseudocounts for intr-cluster transitions,
        by default 0.
    cluster_perturbation_rate : float, optional
        The probability of applying a new permutation between two
        clusters at a given time step in the intial_partition. This
        sampling of two-cycles continues until False is sampled or
        every cluster has been permutated once, by default .1
    """

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
            max_iter=500,
            tol_iter=1e-5,
            n_jobs=-1,
            verbose=1,
            blockmodel_params=None,
            model_id=None,
            debug_list=None,
            debug_output='.',
            # Specific DLBM arguments
            n_iter_supp_smoothing=5,
            parameter_smoothing=None,
            smoothing_schedule=None,
            diag_pi_init=None,
            diag_rho_init=None,
            prior_diagonal_pi=0.,
            prior_diagonal_rho=0.,
            cluster_perturbation_rate=.1,
    ):
        super().__init__(
            Kz=Kz,
            Kw=Kw,
            init_type=init_type,
            em_type=em_type,
            n_init=n_init,
            n_init_clustering=n_init_clustering,
            node_perturbation_rate=node_perturbation_rate,
            model_type=model_type,
            type_init_margins=type_init_margins,
            min_float=min_float,
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

        for arg in ['parameter_smoothing', 'smoothing_schedule']:
            if arg is None:
                raise (f'Argument {arg} for class {type(self).__name__} must be initialized '
                       'explicitly. See documentation for possible values.')

        assert isinstance(parameter_smoothing, bool)
        assert ((smoothing_schedule[1:] - smoothing_schedule[:-1]) > 0).all()
        assert n_iter_supp_smoothing > 0

        self.directed = True  # for compatibility with dsbm

        self.parameter_smoothing = parameter_smoothing
        self.smoothing_schedule = smoothing_schedule if self.parameter_smoothing else []

        self.diag_pi_init = diag_pi_init
        self.diag_rho_init = diag_rho_init
        self.prior_diagonal_pi = prior_diagonal_pi
        self.prior_diagonal_rho = prior_diagonal_rho

        self.n_iter_supp_smoothing = n_iter_supp_smoothing
        self.cluster_perturbation_rate = cluster_perturbation_rate

        self.pi, self.rho = None, None
        self.pi_mask, self.rho_mask = None, None
        self.prior_pi, self.prior_rho = None, None

        # In the dynamic case, if there is margins, they are estimated
        self.estimated_margins = (self.model_type == 'with_margins')
        self.model_parameters = ['log_alpha', 'log_beta', 'log_pi', 'log_rho', 'gamma']
        if self.model_type == 'with_margins':
            self.model_parameters += ['mu', 'nu']

        self.absent_nodes = None

        self.tau = None

        self.density_part_Lc = None

    def fit(
        self, X,
        given_Z=None, given_W=None,
        given_mu=None, given_nu=None
    ):
        """Fits the model to the given data

        Parameters
        ----------
        X : np.ndarray with ndim == 3, first axis representing the
            time, second reprenting the rows and third the columns |
            list of scipy.sparse matrices, where each matrix is a snapshot
            of the graph.
            The discrete-time dynamic bi-partite graph to fit the data to.
        given_Z : np.ndarray of shape (, n_rows) or shape (n_timesteps, n_rows)
            such that each values of the array indicates the cluster of the row.
            The values must be in {0, ..., Kz-1}, as returned by sklearn api
            e.g. KMeans().fit(X).labels_
            This parameter must not be None if the `init_type` parameter is
            `'given'`, otherwise it will not be used. By default None
        given_W : np.ndarray of shape (, n_cols) or shape (n_timesteps, n_cols)
            such that each values of the array indicates the cluster of the row.
            The values must be in {0, ..., Kz-1}, as returned by sklearn api
            e.g. KMeans().fit(X).labels_
            This parameter must not be None if the `init_type` parameter is
            `'given'`, otherwise it will not be used. By default None
        given_mu : np.ndarray of shape (n_timesteps, n_rows), optional
            Gives an initial value for the margin parameter mu of the model.
            This parameter must not be None if the `type_init_margins` parameter is
            `'given'`, otherwise it will not be used. By default None
        given_nu : np.ndarray of shape (n_timesteps, n_cols), optional
            Gives an initial value for the margin parameter nu of the model.
            This parameter must not be None if the `type_init_margins` parameter is
            `'given'`, otherwise it will not be used. By default None

        Returns
        -------
        self : object
            Fitted estimator.
        """
        general.check_X(
            X=X,
            is_graph=False,
            self_loops=True,
            directed=True
        )
        self.X = X
        self.T = len(self.X)
        self.N, self.D = self.X[0].shape
        self.absent_nodes = absent_nodes.AbsentNodes(
            self.X,
            self.threshold_absent_nodes,
            'LBM'
        )
        self._set_transition_masks()
        self._set_global_init_partition(given_Z, given_W)
        self._init_debug()
        self._set_data_margins()

        (all_iter_criterions,
         all_intermediate_iter_criterions,
         all_icls,
         all_row_partitions,
         all_col_partitions) = [], [], [], [], []

        # each initialization of the model
        for init_number in range(self.n_init):
            self._print_verbose_msg_init(init_number)
            self._set_current_init_partition()
            self._init_q()
            self._init_margins(given_mu, given_nu)
            self.tau = 0.  # the smoothing parameter, \in [0., 1.]
            self._full_m_step()
            self._debug(init_number)

            old_iter_criterion = - np.finfo(np.float32).max
            new_iter_criterion = - np.finfo(np.float32).max
            iter_criterions, intermediate_iter_criterions = [], []
            iter_number = 0

            # each iteration of the model, before smoothing
            for _ in range(self.max_iter):
                self._print_verbose_msg_iter(iter_number)
                self._debug(init_number)

                # E + M steps
                old_iter_criterion = new_iter_criterion
                interm_criterion, new_iter_criterion = self._fit_single()

                iter_criterions.append(new_iter_criterion)
                intermediate_iter_criterions.append(interm_criterion)

                self._set_best_parameters(new_iter_criterion, init_number, iter_number)
                delta_iter = general.get_delta(old_iter_criterion, new_iter_criterion)

                if (iter_number >= self.n_iter_min) and (delta_iter < self.tol_iter):
                    self._print_verbose_converged(iter_number, None)
                    break
                iter_number += 1

            self._print_verbose_smoothing(self.smoothing_schedule, self.n_iter_supp_smoothing)

            # each iteration of the model, with smoothing
            for ind_tau, tau in enumerate(self.smoothing_schedule):
                self.tau = tau
                for _ in range(self.n_iter_supp_smoothing):
                    self._debug(init_number)

                    # E + M steps
                    old_iter_criterion = new_iter_criterion
                    interm_criterion, new_iter_criterion = self._fit_single()

                    iter_criterions.append(new_iter_criterion)
                    intermediate_iter_criterions.append(interm_criterion)

                    self._set_best_parameters(new_iter_criterion, init_number, iter_number)
                    delta_iter = general.get_delta(old_iter_criterion, new_iter_criterion)

                    if delta_iter < self.tol_iter:
                        self._print_verbose_converged(iter_number, ind_tau)
                        break
                    iter_number += 1
            if self.verbose >= 1:
                print('Done')

            all_row_partitions.append(self.Z.argmax(axis=2).copy())
            all_col_partitions.append(self.W.argmax(axis=2).copy())

            all_iter_criterions.append(iter_criterions)
            all_intermediate_iter_criterions.append(intermediate_iter_criterions)
            all_icls.append(self.icl())

        self.all_row_partitions = all_row_partitions
        self.all_col_partitions = all_col_partitions
        self.all_iter_criterions = all_iter_criterions
        self.all_intermediate_iter_criterions = all_intermediate_iter_criterions
        self.all_icls = all_icls

        self.fitted = True
        self._set_mixture_proportions()
        self._write_best_parameters()
        return self

    def _fit_single(self):
        """
        a single iteration of EM:
            - computes the log density
            - applies a row E-step
            - applies an M-step on row parameters
            - computes the log density
            - applies a col E-step
            - applies an M-step on col parameters
            - returns a tuple containing the likelihoods
              after row E+M step and after col E+M step
        """
        L1 = self._fit_single_one_sided(mode='row')
        L2 = self._fit_single_one_sided(mode='col')
        return L1, L2

    def _fit_single_one_sided(self, mode):
        """
        a single iteration of EM :
            - applies a row/col E-step
            - applies an M-step on row/col parameters
            - returns the complete data loglikelihood after row/col iteration
        """
        L = 0.
        num_gamma, den_gamma, den_mu, den_nu = self._init_num_den_parameters()

        for t in range(self.T):
            # e step
            X_red_t = self._e_step(t, mode)

            # sufficient statistics for the m step at time t
            (num_gamma[t], den_gamma[t],
             den_mu[t], den_nu[t], Lc_t) = self._m_step_t(t, X_red_t, mode)

            # local part of the loglikelihhod
            L += Lc_t
        self.density_part_Lc = L

        # m step
        self._set_parameters(num_gamma, den_gamma, den_mu, den_nu)
        self._update_mixture_proportions(mode)
        L += self._mixture_part_complete_data_loglikelihood()
        L += self.entropy()
        return L

    def entropy(self):
        """Compute entropy"""
        if self.em_type == 'VEM':
            Hz = m_step.entropy_dynamic(
                self.Z,
                self.qz,
                self.absent_nodes.appearing_row_nodes,
                self.min_float
            )
            Hw = m_step.entropy_dynamic(
                self.W,
                self.qw,
                self.absent_nodes.appearing_col_nodes,
                self.min_float
            )
            return Hz + Hw

        return 0.

    def _m_step_t(self, t, X_red_t, mode):
        """
        Fills the numerators and denominators of
        the parameters of the model at a given time step t

        We fill num_gamma, den_gamma, den_mu, den_nu
        time step by time step. We then smooth theses quantities if needed.

        Note that we keep time variying numerators and denominators
        even if the parameter does not depend on time in the model
        since we compute sufficient statistics at time t to
        estimate gamma, mu and nu

        Note that, contrary to the static case, the complete
        data log likelihood is computed after the E step. In fact,
        otherwise, we would have to compute num_gamma[t] and
        den_gamma[t] for each timestep, to obtain gamma. Then,
        we would have to compute Lc_t for each t, requiring to
        recompute reduced matrices for each time step
        """
        if self.model_type == 'with_margins':
            mu_t, nu_t = self.mu[t], self.nu[t]
        elif self.model_type == 'without_margins':
            mu_t, nu_t = None, None

        # num_gamma_t = X_ZW
        num_gamma_t, den_gamma_t = m_step.update_gamma(
            mode, self.estimated_margins,
            self.X[t], self.Z[t], self.W[t],
            X_red_t, mu_t, nu_t,
            self.dtype, self.model_type
        )
        gamma_no_smooth_t = m_step.get_gamma(
            num_gamma_t,
            den_gamma_t,
            self.em_type,
            self.min_float,
            self.min_gamma
        )
        if self.model_type == 'with_margins':
            den_mu_t = m_step.get_denominator_mu(
                self.Z[t],
                self.W[t],
                self.nu[t],
                gamma_no_smooth_t
            )
            mu_no_smooth_t = self.Xi_[t] / (den_mu_t + self.min_float)
            mu_no_smooth_t[self.absent_nodes.absent_row_nodes[t]] = self.min_float
            den_nu_t = m_step.get_denominator_nu(
                self.Z[t],
                self.W[t],
                mu_no_smooth_t,
                gamma_no_smooth_t
            )

        elif self.model_type == 'without_margins':
            den_mu_t, den_nu_t = None, None

        # in mode 'init', at initialization, there
        # is no self.gamma yet. Ideally, we should first
        # estimate gamma, and then compute Lc, but
        # it is less expensive to compute an approximation
        # of Lc using the local estimate of gamma
        if mode == 'init':
            gamma_t = gamma_no_smooth_t
        else:
            gamma_t = self.gamma if self.model_type == 'with_margins' else self.gamma[t]

        Lc_t = m_step.compute_Lc_static_density(
            self.model_type,
            self.estimated_margins,
            gamma_t,
            mu_t, nu_t,
            self.Xi_[t], self.X_j[t],
            num_gamma_t,
            self.min_float
        )
        return num_gamma_t, den_gamma_t, den_mu_t, den_nu_t, Lc_t

    def _update_pi_rho(self, ZW, qzw, prior, mask):
        if self.em_type == 'VEM':
            pi_rho = m_step.update_pi_rho(ZW, qzw, prior, mask)
        elif self.em_type == 'CEM':
            pi_rho = m_step.update_pi_rho_cem(ZW, prior, mask, self.dtype)

        pi_rho = m_step.correct_pi_rho(
            pi_rho,
            self.min_proba_mixture_proportions,
            self.min_float
        )
        return np.log(pi_rho + self.min_float)

    def _e_step(self, t, mode):
        if self.em_type == 'VEM':
            return self._e_step_vem(t, mode)
        if self.em_type == 'CEM':
            return self._e_step_cem(t, mode)
        raise ValueError(self, "em_type", self.em_type)

    def _e_step_cem(self, t, mode):
        mu_t, nu_t = (self.mu[t], self.nu[t]) if self.model_type == 'with_margins' else (None, None)
        # gamma_t = self.gamma if self.model_type == 'with_margins' else self.gamma[t]
        if mode == 'row':
            self.Z[t], X_red_t = e_step.e_step_t_dynamic_cem(
                mode='row',
                model_type=self.model_type,
                estimated_margins=self.estimated_margins,
                X_t=self.X[t],
                gamma=self.gamma,
                dtype=self.dtype,
                min_float=self.min_float,
                log_pi_rho=self.log_pi,
                log_alpha_beta=self.log_alpha,
                ind_appearing_nodes_t=self.absent_nodes.appearing_row_nodes[t],
                ind_absent_nodes_t=self.absent_nodes.absent_row_nodes[t],
                ZW_tm1=None if t == 0 else self.Z[t - 1],
                ZW_tp1=None if t == (self.T - 1) else self.Z[t + 1],
                Z=None,
                W=self.W[t],
                nu_t=nu_t,
                mu_t=mu_t
            )
        elif mode == 'col':
            self.W[t], X_red_t = e_step.e_step_t_dynamic_cem(
                mode='col',
                model_type=self.model_type,
                estimated_margins=self.estimated_margins,
                X_t=self.X[t],
                gamma=self.gamma,
                dtype=self.dtype,
                min_float=self.min_float,
                log_pi_rho=self.log_rho,
                log_alpha_beta=self.log_beta,
                ind_appearing_nodes_t=self.absent_nodes.appearing_col_nodes[t],
                ind_absent_nodes_t=self.absent_nodes.absent_col_nodes[t],
                ZW_tm1=None if t == 0 else self.W[t - 1],
                ZW_tp1=None if t == (self.T - 1) else self.W[t + 1],
                Z=self.Z[t],
                W=None,
                nu_t=nu_t,
                mu_t=mu_t
            )
        return X_red_t

    def _e_step_vem(self, t, mode):
        """
        Applies an E-step for a given time step t
        on the row or column posterior distributions.
        If t == 0:
            updates the initial posterior proba (ZW[0])
        else:
            in VEM:
                updates the transition posterior proba (qzw[t])
                then updates posterior proba of appearing nodes (ZW_app)
                then updates ZW[t] with qzw[t] and ZW[t-1]
            in CEM:
                computes an unormalized posterior proba vector zw_crit
                then updates ZW[t] with zw_crit and ZW[t-1]
        mode : row or col e-step
        """
        mu_t, nu_t = (self.mu[t], self.nu[t]) if self.model_type == 'with_margins' else (None, None)
        # gamma_t = self.gamma if self.model_type == 'with_margins' else self.gamma[t]
        if mode == 'row':
            self.Z[t], X_red_t = e_step.e_step_t_dynamic_vem(
                mode='row',
                first_ts=(t == 0),
                model_type=self.model_type,
                estimated_margins=self.estimated_margins,
                X_t=self.X[t],
                gamma=self.gamma,
                dtype=self.dtype,
                min_float=self.min_float,
                qzw_tp1=self.qz[t + 1] if (t != self.T - 1) else None,
                log_pi_rho=self.log_pi,
                log_alpha_beta=self.log_alpha,
                ind_appearing_nodes_t=self.absent_nodes.appearing_row_nodes[t],
                ind_absent_nodes_t=self.absent_nodes.absent_row_nodes[t],
                min_proba=self.min_proba_Z,
                ZW_tm1=None if t == 0 else self.Z[t - 1],
                Z=None,
                W=self.W[t],
                nu_t=nu_t,
                mu_t=mu_t
            )
        elif mode == 'col':
            self.W[t], X_red_t = e_step.e_step_t_dynamic_vem(
                mode='col',
                first_ts=(t == 0),
                model_type=self.model_type,
                estimated_margins=self.estimated_margins,
                X_t=self.X[t],
                gamma=self.gamma,
                dtype=self.dtype,
                min_float=self.min_float,
                qzw_tp1=self.qw[t + 1] if (t != self.T - 1) else None,
                log_pi_rho=self.log_rho,
                log_alpha_beta=self.log_beta,
                ind_appearing_nodes_t=self.absent_nodes.appearing_col_nodes[t],
                ind_absent_nodes_t=self.absent_nodes.absent_col_nodes[t],
                min_proba=self.min_proba_W,
                ZW_tm1=None if t == 0 else self.W[t - 1],
                Z=self.Z[t],
                W=None,
                nu_t=nu_t,
                mu_t=mu_t
            )
        return X_red_t

    def _update_mixture_proportions(self, mode):
        """
        Updates the mixture proportions
        """
        if mode == 'init':
            self._update_alpha()
            self._update_beta()
            self._update_pi()
            self._update_rho()
        elif mode == 'row':
            self._update_alpha()
            self._update_pi()
        elif mode == 'col':
            self._update_beta()
            self._update_rho()

    def _set_parameters(self, num_gamma, den_gamma, den_mu, den_nu):
        """
        Updates gamma, mu and nu with their num_... and den_...
        The smoothing is applied here.
        Gamma, mu, nu can be constant or not. Anyway,
        we deal with num_... and den_... as temporal signals
        """
        # gamma
        if self.model_type == 'with_margins':
            self.gamma = m_step.get_gamma(
                num_gamma.sum(0),
                den_gamma.sum(0),
                self.em_type,
                self.min_float,
                self.min_gamma
            )
        elif self.model_type == 'without_margins':
            if self.parameter_smoothing:
                W_tau = m_step.smoothing_matrix(self.T, self.tau, self.dtype)
                num_gamma = num_gamma.reshape((self.T, self.Kz * self.Kw))
                den_gamma = den_gamma.reshape((self.T, self.Kz * self.Kw))

                with np.errstate(under='ignore'):
                    smoothed_num_gamma = (W_tau @ num_gamma).reshape((self.T, self.Kz, self.Kw))
                    smoothed_den_gamma = (W_tau @ den_gamma).reshape((self.T, self.Kz, self.Kw))
                    self.gamma = m_step.get_gamma(
                        smoothed_num_gamma,
                        smoothed_den_gamma,
                        self.em_type,
                        self.min_float,
                        self.min_gamma
                    )
            else:
                self.gamma = m_step.get_gamma(
                    num_gamma,
                    den_gamma,
                    self.em_type,
                    self.min_float,
                    self.min_gamma
                )
        # margins
        if self.model_type == 'with_margins':
            if self.parameter_smoothing:
                W_tau = m_step.smoothing_matrix(self.T, self.tau, self.dtype)
                if self.absent_nodes.n_absent_row_tot > 0:
                    absent_nodes.replace_vals_absent(
                        den_mu,
                        self.absent_nodes.inds_prev_rows,
                        self.absent_nodes.ts_absent_rows
                    )
                if self.absent_nodes.n_absent_col_tot > 0:
                    absent_nodes.replace_vals_absent(
                        den_nu,
                        self.absent_nodes.inds_prev_cols,
                        self.absent_nodes.ts_absent_cols
                    )
                with np.errstate(under='ignore'):
                    smoothed_num_mu = W_tau @ self.Xi_
                    smoothed_den_mu = W_tau @ den_mu
                    self.mu = smoothed_num_mu / (smoothed_den_mu + self.min_float)

                    smoothed_num_nu = W_tau @ self.X_j
                    smoothed_den_nu = W_tau @ den_nu
                    self.nu = smoothed_num_nu / (smoothed_den_nu + self.min_float)
            else:
                self.mu = self.Xi_ / (den_mu + self.min_float)
                self.nu = self.X_j / (den_nu + self.min_float)

            self._correct_margins()

    def _update_alpha(self):
        self.log_alpha = m_step.update_alpha_beta_dynamic(
            self.Z,
            self.absent_nodes.n_absent_row_tot,
            self.absent_nodes.appearing_row_nodes,
            self.absent_nodes.absent_row_nodes[0],
            self.min_float,
            self.min_proba_mixture_proportions,
            self.dtype
        )

    def _update_beta(self):
        self.log_beta = m_step.update_alpha_beta_dynamic(
            self.W,
            self.absent_nodes.n_absent_col_tot,
            self.absent_nodes.appearing_col_nodes,
            self.absent_nodes.absent_col_nodes[0],
            self.min_float,
            self.min_proba_mixture_proportions,
            self.dtype
        )

    def _update_pi(self):
        self.log_pi = self._update_pi_rho(
            self.Z,
            self.qz,
            self.prior_pi,
            self.pi_mask
        )

    def _update_rho(self):
        self.log_rho = self._update_pi_rho(
            self.W,
            self.qw,
            self.prior_rho,
            self.rho_mask
        )

    def _correct_margins(self):
        """
        Sets margins to zero for absent nodes.
        Useful if smoothing is applied

        It should be noted that num_mu and den_mu are 0 for absent nodes.
        When the smoothing is applied, the margin of
        an absent node does not contribute to the
        other margins. Moreover, for an absent node,
        the smoothed margin can be non zero : it must then
        be corrected.
        """
        np.clip(self.mu, self.min_margin, None, self.mu)
        np.clip(self.nu, self.min_margin, None, self.nu)

        # important
        if self.model_type == 'with_margins':
            for t in range(self.T):
                self.mu[t][self.absent_nodes.absent_row_nodes[t]] = self.min_float
                self.nu[t][self.absent_nodes.absent_col_nodes[t]] = self.min_float

    def _init_num_den_parameters(self):
        """
        Initializes the time-dependant vectors
        that will be filled in _m_step_t()
        """
        num_gamma = np.zeros((self.T, self.Kz, self.Kw), dtype=self.dtype)
        den_gamma = np.zeros((self.T, self.Kz, self.Kw), dtype=self.dtype)

        # we initialize den_mu, den_nu even if the model
        # is without margins, for consistency
        den_mu = np.zeros((self.T, self.N), dtype=self.dtype)
        den_nu = np.zeros((self.T, self.D), dtype=self.dtype)

        return num_gamma, den_gamma, den_mu, den_nu

    def _init_q(self):
        """
        Initializes the variational probability distributions:
            qz, Z, Z_app, qw, W, W_app,
        where qz and qw are initialized only in VEM
        """
        if self.em_type == 'VEM':
            self.Z = np.zeros((self.T, self.N, self.Kz), dtype=self.dtype)
            self.Z[0] = init.init_ZW0(
                self.Kz,
                self.current_init_Z[0],
                self.min_proba_Z,
                self.dtype
            )
            self.qz = init.init_qzw(
                self.Kz,
                self.current_init_Z,
                self.min_proba_Z,
                self.dtype
            )
            for t in range(1, self.T):
                self.Z[t] = e_step.update_ZW_t_vem(
                    self.Z[t - 1],
                    self.qz[t - 1],
                    None, None, None,
                    self.min_proba_Z
                )
            self.Z = e_step.correct_ZW(self.Z, self.min_proba_Z)

            self.W = np.zeros((self.T, self.D, self.Kw), dtype=self.dtype)
            self.W[0] = init.init_ZW0(
                self.Kw,
                self.current_init_W[0],
                self.min_proba_W,
                self.dtype
            )
            self.qw = init.init_qzw(
                self.Kw,
                self.current_init_W,
                self.min_proba_W,
                self.dtype
            )
            for t in range(1, self.T):
                self.W[t] = e_step.update_ZW_t_vem(
                    self.W[t - 1],
                    self.qw[t - 1],
                    None, None, None,
                    self.min_proba_W
                )
            self.W = e_step.correct_ZW(self.W, self.min_proba_W)

        elif self.em_type == 'CEM':
            self.qz = None
            self.Z = np.zeros((self.T, self.N, self.Kz), dtype='bool')
            for t in range(self.T):
                self.Z[t] = init.init_ZW0(
                    self.Kz,
                    self.current_init_Z[t],
                    None, 'bool'
                )
            self.W = np.zeros((self.T, self.D, self.Kw), dtype='bool')
            for t in range(self.T):
                self.W[t] = init.init_ZW0(
                    self.Kw,
                    self.current_init_W[t],
                    None, 'bool'
                )
            self.qw = None

    def _set_current_init_partition(self):
        """
        Creates the current init partition by
        applying noise to the global init partition
        """
        self.current_init_Z = init.apply_perturbation_to_init_clustering(
            init_partition=self.global_init_Z,
            random_state=self.random_state,
            node_perturbation_rate=self.node_perturbation_rate,
            cluster_perturbation_rate=self.cluster_perturbation_rate,
            dynamic=True
        )
        self.current_init_W = init.apply_perturbation_to_init_clustering(
            init_partition=self.global_init_W,
            random_state=self.random_state,
            node_perturbation_rate=self.node_perturbation_rate,
            cluster_perturbation_rate=self.cluster_perturbation_rate,
            dynamic=True
        )

    def _set_global_init_partition(self, given_Z, given_W):
        """
        Sets the global row and column partitions of the model.
        At each new initialization of the model, noise will be applied to
        this partition to create the current init partition
        """
        # wether we convert the obtained (N) partition to a T x N matrix
        if given_Z is not None or given_W is not None:
            assert (given_Z is not None) and (given_W is not None)
            assert self.init_type == 'given'
            assert given_Z.ndim == given_W.ndim

        # X_init : data matrix used for initial clustering
        X_init_row = init.get_X_init(
            self.X,
            mode='row',
            absent_nodes=self.absent_nodes.absent_row_nodes
        )
        self.global_init_Z = init.get_init_partition(
            X_init=X_init_row,
            Kzw=self.Kz,
            init_type=self.init_type,
            T=self.T,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            n_init=self.n_init_clustering,
            given_partition=given_Z
        )
        X_init_col = init.get_X_init(
            self.X,
            mode='col',
            absent_nodes=self.absent_nodes.absent_col_nodes
        )
        self.global_init_W = init.get_init_partition(
            X_init=X_init_col,
            Kzw=self.Kw,
            init_type=self.init_type,
            T=self.T,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            n_init=self.n_init_clustering,
            given_partition=given_W
        )

    def _mixture_part_complete_data_loglikelihood(self):
        """
        Computes the terms in alpha, beta, rho and pi
        of the log likelihood of the expected complete data
        """
        # alpha and beta part
        alpha_part = m_step.complete_data_loglikelihood_alpha_beta(
            self.log_alpha,
            self.Z,
            self.absent_nodes.absent_row_nodes[0],
            self.absent_nodes.appearing_row_nodes
        )
        beta_part = m_step.complete_data_loglikelihood_alpha_beta(
            self.log_beta,
            self.W,
            self.absent_nodes.absent_col_nodes[0],
            self.absent_nodes.appearing_col_nodes
        )
        pi_part = m_step.complete_data_loglikelihood_pi_rho(
            self.em_type,
            self.log_pi,
            self.Z, self.qz,
            self.pi_mask
        )
        rho_part = m_step.complete_data_loglikelihood_pi_rho(
            self.em_type,
            self.log_rho,
            self.W, self.qw,
            self.rho_mask
        )
        return alpha_part + beta_part + pi_part + rho_part

    def _set_data_margins(self):
        if isinstance(self.X, list):
            self.Xi_ = np.array([self.X[t].sum(1).A1 for t in range(self.T)])
            self.X_j = np.array([self.X[t].sum(0).A1 for t in range(self.T)])
        else:
            self.Xi_ = self.X.sum(2)
            self.X_j = self.X.sum(1)

        if self.absent_nodes.n_absent_row_tot > 0:
            absent_nodes.replace_vals_absent(
                self.Xi_,
                self.absent_nodes.inds_prev_rows,
                self.absent_nodes.ts_absent_rows
            )
        if self.absent_nodes.n_absent_col_tot > 0:
            absent_nodes.replace_vals_absent(
                self.X_j,
                self.absent_nodes.inds_prev_cols,
                self.absent_nodes.ts_absent_cols
            )

    def _init_margins(self, given_mu=None, given_nu=None):
        """
        Initializes the margins mu and nu, that
        can not be initialized from a given partition
        as it is the case for alpha, pi, gamma, ...
        """
        # time dependant margins
        if self.model_type == 'with_margins':
            if self.type_init_margins == 'ones':
                self.mu = np.ones((self.T, self.N), dtype=self.dtype)
                self.nu = np.ones((self.T, self.D), dtype=self.dtype)
            elif self.type_init_margins == 'X.':
                self.mu = self.Xi_.copy()
                self.nu = self.X_j.copy()
            elif self.type_init_margins == 'random':
                self.mu = np.abs(self.random_state.normal(
                    loc=self.loc_random_margins,
                    scale=self.scale_random_margins,
                    size=(self.T, self.N)
                ).astype(self.dtype))
                self.nu = np.abs(self.random_state.normal(
                    loc=self.loc_random_margins,
                    scale=self.scale_random_margins,
                    size=(self.T, self.D)
                ).astype(self.dtype))
            elif self.type_init_margins == 'given':
                assert given_mu is not None and given_nu is not None
                self.mu = given_mu.astype(self.dtype)
                self.nu = given_nu.astype(self.dtype)

    def _set_transition_masks(self):
        # masks used to update pi and rho efficiently
        # by only selecting present and non-appearing nodes
        self.pi_mask = init.pi_rho_update_mask(
            self.T, self.N,
            self.absent_nodes.absent_row_nodes,
            self.absent_nodes.appearing_row_nodes
        )
        self.rho_mask = init.pi_rho_update_mask(
            self.T, self.D,
            self.absent_nodes.absent_col_nodes,
            self.absent_nodes.appearing_col_nodes
        )

    def _set_mixture_proportions(self):
        self.alpha = np.exp(self.log_alpha)
        self.beta = np.exp(self.log_beta)
        self.pi = np.exp(self.log_pi)
        self.rho = np.exp(self.log_rho)

    def set_prior_transition_matrices(self):
        """
        For pi and rho, we consider an informatiVe dirichlet prior
        for each row of the transition matrices.
        The prior is Kzw x Kzw and st prior[k, l] = 1
        and prior[k, k] = prior_diagonal > 1.
        Its role is to favor partitions with less class transitions.
        During the M-step, it adds the pseudocounts, with eg
        pi_kl propto n[k, l] + prior[k, l] - 1
        where n[k, l] is the number of class transitions, as in
        a classical M step.
        """
        assert 0. <= self.prior_diagonal_pi <= 1.
        assert 0. <= self.prior_diagonal_rho <= 1.

        prior_pi_val = self.prior_diagonal_pi * self.N + 1
        prior_rho_val = self.prior_diagonal_rho * self.D + 1

        IZ = np.eye((self.Kz), dtype=self.dtype)
        OZ = np.ones((self.Kz, self.Kz), dtype=self.dtype)
        self.prior_pi = (prior_pi_val - 1.) * IZ + OZ

        IW = np.eye((self.Kw), dtype=self.dtype)
        OW = np.ones((self.Kw, self.Kw), dtype=self.dtype)
        self.prior_rho = (prior_rho_val - 1.) * IW + OW

    def _full_m_step(self):
        """
        Applies an M-step to all the current parameters.
        Used after the initialization of the posterior probabilities.
        """
        # update all mixture prop
        self.set_prior_transition_matrices()
        self._update_mixture_proportions(mode='init')

        # update gamma, mu, nu
        # by filling num_mu, num_nu, num_gamma, den...
        num_gamma, den_gamma, den_mu, den_nu = self._init_num_den_parameters()

        for t in range(self.T):
            (num_gamma[t], den_gamma[t],
             den_mu[t], den_nu[t], _) = self._m_step_t(t=t, X_red_t=None, mode='init')

        # now that the num and den of the params have
        # been set, we set the params values (with or without smoothing)
        self._set_parameters(num_gamma, den_gamma, den_mu, den_nu)

    def _init_pi_rho(self):
        """
        Initializes pi and rho, the transition matrices,
        with a M step or with a matrix with diagonal
        value diag_pi_rho_init
        """
        if self.diag_pi_init is not None:
            val_off_diag = (1. - self.diag_pi_init) / (self.Kz - 1)
            OZ = np.full((self.Kz, self.Kz), val_off_diag)
            IZ = np.eye(self.Kz)
            pi = OZ + (self.diag_pi_init - val_off_diag) * IZ
            self.log_pi = np.log(pi)
        else:
            self._update_pi()

        if self.diag_rho_init is not None:
            val_off_diag = (1. - self.diag_rho_init) / (self.Kw - 1)
            OW = np.full((self.Kw, self.Kw), val_off_diag)
            IW = np.eye(self.Kw)
            rho = OW + (self.diag_rho_init - val_off_diag) * IW
            self.log_rho = np.log(rho)
        else:
            self._update_rho()

    def icl(self):
        """Compute ICL"""
        icl_z = general.compute_mixture_exact_icl(self.Z, self.Kz, self.N)
        icl_w = general.compute_mixture_exact_icl(self.W, self.Kw, self.D)
        bic_penalty = (
            -.5 * (self.Kz * self.Kw + self.T * (self.N + self.D)) *
            np.log(self.T * self.N * self.D)
        )
        val = (
            icl_z + icl_w +
            self.density_part_Lc +
            bic_penalty
        )
        return val

    def best_partition(self, mode, n_first=1):
        """
        returns a list of tuple of partitions
            if 1 partition: [(part_1), ..., (part_n_first)]
            if 2 partitions: [(row_part_1, col_part_1), ...,
                                (row_part_n_first, col_part_n_first)]

        mode == 'likelihood' returns the n_first best partitions
        in terms of likelihood

        if mode == 'consensus: hbgf' or mode == 'consensus: cspa'
         returns 1 consensus partition over n_first partitions
        """
        assert self.fitted
        best_partitions = metrics.sort_partitions(
            self.all_iter_criterions,
            [self.all_row_partitions, self.all_col_partitions],
            n_first
        )
        # put absent nodes in cluster -1
        for p1, p2 in best_partitions:
            for t in range(self.T):
                for i in self.absent_nodes.absent_row_nodes[t]:
                    p1[t, i] = -1
                for j in self.absent_nodes.absent_col_nodes[t]:
                    p2[t, j] = -1

        if mode == 'likelihood':
            return best_partitions
        if mode in ['consensus: hbgf', 'consensus: cspa']:
            best_row_partitions = [x[0] for x in best_partitions]
            best_col_partitions = [x[1] for x in best_partitions]
            if mode == 'consensus: hbgf':
                Z_consensus = consensus.hbgf(
                    best_row_partitions,
                    self.n_init_clustering_consensus
                ).reshape(self.T, self.N)
                W_consensus = consensus.hbgf(
                    best_col_partitions,
                    self.n_init_clustering_consensus
                ).reshape(self.T, self.D)
            elif mode == 'consensus: cspa':
                Z_consensus = consensus.cspa(
                    best_row_partitions, self.n_init_clustering_consensus
                ).reshape(self.T, self.N)
                W_consensus = consensus.cspa(
                    best_col_partitions,
                    self.n_init_clustering_consensus
                ).reshape(self.T, self.D)
            return [(Z_consensus, W_consensus)]
        raise ValueError
