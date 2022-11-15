"""E-step-related general functions"""

import numpy as np

from . import general

nax = np.newaxis


def log_density_t(
    X_t, gamma, mode,
    model_type, estimated_margins, dtype, min_float,
    Z=None, W=None,
    mu_t=None, nu_t=None
):
    """
    deals with static and dynamic cases
    """
    log_gamma = np.log(gamma + min_float)
    if mode == 'row':
        X_W = (X_t @ W).astype(dtype)
        dens = X_W @ log_gamma.T
        if model_type in ['without_margins', 'without_margins+free_gamma']:
            W_ = general.get_class_counts(W).astype(dtype)
            np.add(dens, - (gamma @ W_.T)[nax, :], out=dens)
        elif ((model_type == 'with_margins' and estimated_margins)
                or model_type == 'free_margins+constant_gamma'):
            nu_t_W = (nu_t @ W).astype(dtype)
            np.add(dens, - mu_t[:, nax] * (gamma @ nu_t_W.T)[nax, :], out=dens)
        X_red = X_W

    elif mode == 'col':
        X_Z = (X_t.T @ Z).astype(dtype)
        dens = X_Z @ log_gamma
        if model_type in ['without_margins', 'without_margins+free_gamma']:
            Z_ = general.get_class_counts(Z).astype(dtype)
            np.add(dens, - (Z_ @ gamma)[nax, :], out=dens)
        elif ((model_type == 'with_margins' and estimated_margins)
                or model_type == 'free_margins+constant_gamma'):
            mu_t_Z = (mu_t @ Z).astype(dtype)
            np.add(dens, - nu_t[:, nax] * (mu_t_Z @ gamma)[nax, :], out=dens)
        X_red = X_Z
    return dens, X_red


def _get_ZW_t_app(
        dens,
        em_type,
        ind_appearing_nodes_t,
        qzw_tp1,
        ZW_tm1,
        ZW_tp1,
        log_alpha_beta,
        log_pi_rho,
        min_float,
        min_proba
):
    appearing = len(ind_appearing_nodes_t) > 0
    if not appearing:
        return None

    ND = dens.shape[0]
    dens_app = dens[ind_appearing_nodes_t, :]
    if em_type == 'VEM':
        log_prop_app = _get_log_prop_dynamic_vem(
            'appearing',
            qzw_tp1,
            ND,
            log_pi_rho,
            log_alpha_beta,
            min_float
        )
        ZW_app = dens_app + log_prop_app[ind_appearing_nodes_t, :]
        ZW_app = _normalize_posterior(
            ZW_app,
            min_proba,
            ind_absent_nodes_t=None,
            dynamic=False
        )
    elif em_type == 'CEM':
        log_prop_app = _get_log_prop_dynamic_cem(
            ZW_tm1, ZW_tp1,
            log_pi_rho, log_alpha_beta
        )
        ZW_app = dens_app + log_prop_app[ind_appearing_nodes_t, :]
        ZW_app = _ce_step_static(ZW_app)
    return ZW_app


def _get_log_prop_dynamic_vem(
    mode,
    qzw_tp1,
    ND,
    log_pi_rho,
    log_alpha_beta,
    min_float
):
    """
    """
    if qzw_tp1 is not None:
        # dkl_tp1[i, k]
        dkl_tp1 = (qzw_tp1 * (np.log(qzw_tp1 + min_float) - log_pi_rho[nax, :, :])).sum(2)

        if mode in ['appearing', 't0']:
            log_prop = log_alpha_beta[nax, :] - dkl_tp1
        else:
            log_prop = log_pi_rho[nax, :, :] - dkl_tp1[:, :, nax]
    else:
        # case t = T
        if mode in ['appearing', 't0']:
            log_prop = np.tile(log_alpha_beta[nax, :], (ND, 1))
        else:
            log_prop = log_pi_rho[nax, :, :]
    return log_prop


def e_step_t_dynamic_vem(
        mode, first_ts, model_type, estimated_margins,
        X_t, gamma,
        dtype, min_float,
        qzw_tp1, log_pi_rho, log_alpha_beta,
        ind_appearing_nodes_t, ind_absent_nodes_t,
        min_proba, ZW_tm1,
        Z=None, W=None, nu_t=None, mu_t=None
):
    """
    E step at time t on either column or row for VEM
    """
    dens, X_red = log_density_t(
        X_t, gamma, mode,
        model_type, estimated_margins,
        dtype, min_float,
        Z=Z, W=W, nu_t=nu_t, mu_t=mu_t
    )
    ZW_t_app = _get_ZW_t_app(
        dens,
        'VEM',
        ind_appearing_nodes_t,
        qzw_tp1,
        None,
        None,
        log_alpha_beta,
        log_pi_rho,
        min_float,
        min_proba
    )
    log_prop_mode = 't0' if first_ts else 'regular'
    ND = X_red.shape[0]

    log_prop = _get_log_prop_dynamic_vem(
        log_prop_mode,
        qzw_tp1,
        ND,
        log_pi_rho, log_alpha_beta,
        min_float
    )
    if first_ts:
        ZW = log_prop + dens
        ZW = _normalize_posterior(
            ZW, min_proba,
            ind_absent_nodes_t=ind_absent_nodes_t,
            dynamic=False
        )
    else:
        qzw_t = log_prop + dens[:, nax, :]
        qzw_t = _normalize_posterior(
            qzw_t, min_proba,
            ind_absent_nodes_t=ind_absent_nodes_t,
            dynamic=True
        )
        ZW = update_ZW_t_vem(
            ZW_tm1,
            qzw_t,
            ind_appearing_nodes_t,
            ZW_t_app,
            ind_absent_nodes_t,
            min_proba
        )
    return ZW, X_red


def e_step_t_dynamic_cem(
        mode, model_type, estimated_margins,
        X_t, gamma,
        ZW_tm1, ZW_tp1,
        dtype, min_float,
        log_pi_rho, log_alpha_beta,
        ind_appearing_nodes_t, ind_absent_nodes_t,
        Z=None, W=None, nu_t=None, mu_t=None
):
    """
    E step at time t on either column or row for CEM
    """
    dens, X_red = log_density_t(
        X_t, gamma, mode,
        model_type, estimated_margins,
        dtype, min_float,
        Z=Z, W=W, nu_t=nu_t, mu_t=mu_t
    )
    ZW_t_app = _get_ZW_t_app(
        dens,
        'CEM',
        ind_appearing_nodes_t,
        None,
        ZW_tm1,
        ZW_tp1,
        log_alpha_beta,
        log_pi_rho,
        min_float,
        None
    )
    log_prop = _get_log_prop_dynamic_cem(
        ZW_tm1, ZW_tp1,
        log_pi_rho, log_alpha_beta
    )
    ZW_t = log_prop + dens
    ZW_t = _ce_step_dynamic_t(
        ZW_t,
        ind_appearing_nodes_t,
        ZW_t_app,
        ind_absent_nodes_t
    )
    return ZW_t, X_red


def update_ZW_t_vem(
        ZW_tm1,
        qzw_t,
        ind_appearing_t,
        ZW_t_app,
        ind_absent_nodes_t,
        min_proba
):
    """
    marginal posterior probas
    ZW, qzw and ZW_t_app must be proba not log proba
    """
    # here ZW_t are the posterior probabilities
    # at the previous iteration of EM
    with np.errstate(under='ignore'):
        ZW_t = (ZW_tm1[:, :, nax] * qzw_t).sum(axis=1)

    # we replace the previous calculations for
    # the appearing nodes with ZW_t_app
    if ZW_t_app is not None:
        ZW_t[ind_appearing_t] = ZW_t_app

    # to avoid numerical issues
    ZW_t[ind_absent_nodes_t] = 1.

    # normalize
    np.divide(ZW_t, ZW_t.sum(1, keepdims=True), ZW_t)
    np.clip(ZW_t, min_proba, None, ZW_t)
    np.divide(ZW_t, ZW_t.sum(1, keepdims=True), ZW_t)

    # set zero proba for absent nodes
    # so that when we compute Z.T @ X @ W
    # the absent nodes are not counted
    # which is necessary to correctly update
    # gamma (in case thr_absent_nodes != 0)
    # and to compute the complete data log
    # likelihood
    ZW_t[ind_absent_nodes_t] = 0.
    return ZW_t


def e_step_static(
    log_alpha_beta,
    regularize,
    regularization_mode,
    lambda_, S, P,
    damping_factor,
    em_type,
    X, gamma, mode,
    model_type, estimated_margins,
    dtype, min_float, min_proba,
    Z, W, nu, mu
):
    """
    E step on either column or row in static mode
    """
    ZW = Z if mode == 'row' else W
    log_prop = _get_log_prop_static(
        log_alpha_beta,
        regularize,
        regularization_mode,
        lambda_, S, P, ZW
    )
    dens, X_red = log_density_t(
        X, gamma, mode,
        model_type, estimated_margins,
        dtype, min_float,
        Z=Z, W=W, nu_t=nu, mu_t=mu
    )
    ZW_new = log_prop + dens

    if em_type == 'VEM':
        ZW_new = _normalize_posterior(
            ZW_new,
            min_proba,
            ind_absent_nodes_t=None,
            dynamic=False
        )
        ZW_new = _apply_damping(ZW, ZW_new, damping_factor)
    elif em_type == 'CEM':
        ZW_new = _ce_step_static(ZW_new)
        ZW_new = general.to_sparse(ZW_new)

    return ZW_new, X_red


def _normalize_posterior(
    qzw_ZW, min_proba,
    ind_absent_nodes_t=None, dynamic=False
):
    """
    qzw_ZW : could be qzw or ZW
    """
    axis = 2 if dynamic else 1
    max_log = np.max(qzw_ZW, axis=axis, keepdims=True)
    with np.errstate(divide='ignore', under='ignore'):
        np.exp(qzw_ZW - max_log, qzw_ZW)
        np.divide(qzw_ZW, qzw_ZW.sum(axis=axis, keepdims=True), qzw_ZW)

    # only fill un informative values
    # for absent nodes to avoid numerical isssues
    if ind_absent_nodes_t is not None:
        qzw_ZW[ind_absent_nodes_t] = 1.

    np.clip(qzw_ZW, min_proba, None, qzw_ZW)
    with np.errstate(divide='ignore', under='ignore'):
        np.divide(qzw_ZW, qzw_ZW.sum(axis=axis, keepdims=True), qzw_ZW)

    return qzw_ZW


def _apply_damping(ZW_old, ZW_new, damping_factor):
    if damping_factor is None:
        return ZW_new
    return damping_factor * ZW_new + (1. - damping_factor) * ZW_old


def _get_log_prop_static(
        log_alpha_beta,
        regularize,
        regularization_mode,
        lambda_, S, P, ZW
):
    """
    returns a ND x Kzw array containing the log proportions
    that is, for instance for the rows, log_alpha[i, k] if there
    is no regularization, or S_r[i].dot(Z[:, k]) + log_alpha[i, k]
    if there is regularization
    """
    if regularize:
        reg = general.to_dense(lambda_ * (S @ ZW))
        if regularization_mode == 'mixture':
            not_P = ~ P
            log_prop = not_P[:, nax] * log_alpha_beta[nax, :] + P[:, nax] * reg
        elif regularization_mode == 'all':
            log_prop = log_alpha_beta[nax, :] + reg
    else:
        log_prop = log_alpha_beta[nax, :]

    return log_prop


def _ce_step_dynamic_t(crit, ind_appearing_t, ZW_t_app, ind_absent_nodes_t):
    """
    Computes the Classification E step in dynamic
    where z_i = argmax_k crit
    """
    indexes = np.asarray(crit.argmax(1)).T
    ZW_t_new = np.zeros_like(crit, dtype='bool')
    ND = crit.shape[0]
    ZW_t_new[np.arange(ND), indexes] = True

    # we replace the previous calculations for
    # the appearing nodes with ZW_t_app
    if ZW_t_app is not None:
        ZW_t_new[ind_appearing_t] = ZW_t_app

    # put constant proba for absent nodes
    # for we do not care
    ZW_t_new[ind_absent_nodes_t] = False

    return ZW_t_new


def _ce_step_static(crit):
    """
    Computes the Classification E step in static
    where z_i = argmax_k crit
    """
    indexes = np.asarray(crit.argmax(1)).T
    ZW_t_new = np.zeros_like(crit, dtype='bool')
    ZW_t_new[np.arange(crit.shape[0]), indexes] = True
    return ZW_t_new


def _get_log_prop_dynamic_cem(
    ZW_tm1, ZW_tp1,
    log_pi_rho, log_alpha_beta
):
    """
    """
    ND = ZW_tm1.shape[0] if ZW_tm1 is not None else ZW_tp1.shape[0]
    log_pi_rho = np.tile(log_pi_rho[nax], (ND, 1, 1))
    arange = np.arange(ND)

    if ZW_tm1 is not None:
        ind_tm1 = ZW_tm1.argmax(1)
        log_pi_prev = log_pi_rho[arange, ind_tm1, :]
        log_prop = log_pi_prev
    else:
        # case t == 1
        log_prop = log_alpha_beta[nax]

    if ZW_tp1 is not None:
        ind_tp1 = ZW_tp1.argmax(1)
        log_pi_next = log_pi_rho[arange, :, ind_tp1]
        log_prop = log_prop + log_pi_next
    else:
        # case t=T
        pass

    return log_prop


def correct_ZW(ZW, min_proba):
    """
    ZW must be a proba (not a log proba)
    """
    np.clip(ZW, min_proba, None, ZW)
    with np.errstate(divide='ignore', under='ignore'):
        np.divide(ZW, ZW.sum(axis=2, keepdims=True), ZW)
    return ZW
