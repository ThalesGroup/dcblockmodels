"""M-step-related general functions"""

import numpy as np

from . import general

nax = np.newaxis


def update_gamma(
    mode, estimated_margins,
    X, Z, W, X_red, mu, nu,
    dtype, model_type
):
    """
    In the static case or for parameters at a given time step

    Note that here, mu and nu can be the estimated margins or
    the non estimated margins xi_, x_j

    X_red is the reduced matrix (X_W or X_Z), that has been computed
    in the e step (row or col), where row or col is
    determined by the 'mode' keyword.

    For the first m step, at initialization,
    there has not been any e step
    previously, so the computation is different, which
    is indicated by mode == 'init'
    """
    # computes the numerator and denominator of gamma
    if mode == 'init':
        X_kl = general.to_dense(Z.T @ X @ W)
        if model_type == 'with_margins':
            mu_Z = Z.T @ mu
            nu_W = W.T @ nu
            margins_kl = np.outer(mu_Z, nu_W).astype(dtype)

    elif mode == 'row':
        X_W = X_red
        X_kl = general.to_dense(Z.T @ X_W)
        if model_type == 'with_margins':
            mu_Z = Z.T @ mu
            if estimated_margins:
                nu_W = W.T @ nu
            else:
                nu_W = X_W.sum(0)
            nu_W = general.to_dense(nu_W)
            margins_kl = np.outer(mu_Z, nu_W)
    elif mode == 'col':
        X_Z = X_red
        X_kl = general.to_dense(X_Z.T @ W)
        if model_type == 'with_margins':
            nu_W = W.T @ nu
            if estimated_margins:
                mu_Z = Z.T @ mu
            else:
                mu_Z = X_Z.sum(0)
            mu_Z = general.to_dense(mu_Z)
            margins_kl = np.outer(mu_Z, nu_W)

    if model_type == 'without_margins':
        Z_ = general.get_class_counts(Z)
        W_ = general.get_class_counts(W)
        ZW_kl = np.outer(Z_, W_)
        den_gamma = ZW_kl
    elif model_type == 'with_margins':
        den_gamma = margins_kl

    return X_kl, den_gamma


def get_gamma(num_gamma, den_gamma, em_type, min_float, min_gamma):
    """Get usable gamma depending on EM type, with minimum values for denominator & gamma itself"""
    gamma = num_gamma / (den_gamma + min_float)
    if em_type == 'CEM':
        # very important in CEM to avoid empty clusters
        np.clip(gamma, min_gamma, None, gamma)
    return gamma


def get_denominator_mu(Z, W, nu, gamma):
    """Compute denominator of mu"""
    nu_W = W.T @ nu
    den_mu = Z @ gamma @ nu_W
    return den_mu


def get_denominator_nu(Z, W, mu, gamma):
    """Compute denominator of nu"""
    mu_Z = Z.T @ mu
    den_nu = W @ gamma.T @ mu_Z
    return den_nu


def update_pi_rho(ZW, qzw, prior, mask):
    """
    Update pi/rho in VEM mode
    """
    res = prior - 1. + (
        ZW[:-1, :, :, nax] *
        qzw[1:, :, :, :] *
        mask[:, :, nax, nax]
    ).sum((0, 1))
    return res


def correct_pi_rho(mat, min_proba, min_float):
    """
    for pi/rho
    """
    np.divide(mat, mat.sum(axis=1, keepdims=True) + min_float, mat)
    np.clip(mat, min_proba, None, mat)
    np.divide(mat, mat.sum(axis=1, keepdims=True) + min_float, mat)
    return mat


def update_alpha_beta_dynamic(
    ZW, n_absent_nodes, appearing_nodes,
    absent_nodes_t0, min_float, min_proba, dtype
):
    """
    Update alpha/beta in dynamic mode
    """
    if n_absent_nodes > 0:
        present_t0 = np.setdiff1d(
            np.arange(ZW.shape[1]),
            absent_nodes_t0
        )
        res = ZW[0][present_t0].sum(0).astype(dtype)

        T = len(appearing_nodes)
        for t in range(1, T):
            np.add(res, ZW[t][appearing_nodes[t]].sum(0).astype(dtype), res)
    else:
        res = ZW[0].sum(0).astype(dtype)

    np.divide(res, res.sum(), res)
    np.clip(res, min_proba, None, res)
    np.divide(res, res.sum(), res)
    return np.log(res + min_float)


def update_pi_rho_cem(ZW, prior, mask, dtype):
    """
    Update pi/rho in CEM mode
    """
    pi_rho = (prior - 1.) + (
        ZW[:-1, :, :, nax] *
        ZW[1:, :, nax, :] *
        mask[:, :, nax, nax]
    ).sum((0, 1)).astype(dtype)
    return pi_rho


def smoothing_matrix(T, tau, dtype):
    """
    Return the weights used for temporal smoothing
    """
    if tau == 0:
        return np.ones((T, T), dtype=dtype) / T
    if tau == 1:
        return np.eye((T), dtype=dtype)

    arr = np.arange(T)
    t = arr[:, nax]
    t_ = arr[nax, :]
    delta = (t_ - t).astype(dtype)

    with np.errstate(under='ignore'):
        # W = np.exp(- (tau / (1 - tau)) * delta ** 2)
        W = np.exp(- (1. / ((1. / tau) - 1.)) * delta ** 2)
        np.divide(W, W.sum(axis=1, keepdims=True), W)
    return W


def compute_Lc_static_density(
        model_type, estimated_margins,
        gamma, mu, nu, Xi_, X_j,
        X_kl, min_float
):
    """
    Computes the terms in gamma, mu,nu in complete data log likelihood
    """
    Lc = 0.
    log_gamma = np.log(gamma + min_float)
    if model_type == 'without_margins':
        Lc += (X_kl * log_gamma).sum()
    elif model_type == 'with_margins':
        if estimated_margins:
            log_mu = np.log(mu + min_float)
            log_nu = np.log(nu + min_float)
            Lc += (
                (X_kl * log_gamma).sum() +
                Xi_.dot(log_mu) + X_j.dot(log_nu)
            )
        else:
            Lc += (X_kl * log_gamma).sum()
    return Lc


def complete_data_loglikelihood_alpha_beta(
        log_alpha_beta, ZW,
        absent_nodes_t0,
        appearing_nodes
):
    """
    Computes the terms in alpha or beta of the log likelihood of the expected complete data
    """
    T = len(ZW)
    ND = ZW[0].shape[0]

    present_t0 = np.setdiff1d(
        np.arange(ND),
        absent_nodes_t0
    )
    nzw = ZW[0][present_t0].sum(0)

    for t in range(T):
        np.add(nzw, ZW[t][appearing_nodes[t], :].sum(), out=nzw)

    return nzw.dot(log_alpha_beta)


def complete_data_loglikelihood_pi_rho(
        em_type,
        log_pi_rho,
        ZW, qzw,
        mask
):
    """
    Computes the terms in pi or rho of the log likelihood of the expected complete data
    """
    if em_type == 'VEM':
        res = (
            ZW[:-1, :, :, nax] *
            qzw[1:, :, :, :] *
            log_pi_rho[nax, nax, :, :]
        ).sum((2, 3))

    elif em_type == 'CEM':
        res = (
            ZW[:-1, :, :, nax] *
            ZW[1:, :, nax, :] *
            log_pi_rho[nax, nax, :, :]
        ).sum((2, 3))

    return res.sum(where=mask)


def get_regularization(
        regularize_row, regularize_col, lambda_r, lambda_c, S_r, S_c, Z, W):
    """
    Returns the regularization term : lambda_c R_c + lambda_r R_r
    """
    regularization = 0.
    if regularize_row:
        regularization += (.5 * lambda_r * (S_r.sum() - S_r.multiply(Z @ Z.T).sum()))
    if regularize_col:
        regularization += (.5 * lambda_c * (S_c.sum() - S_c.multiply(W @ W.T).sum()))
    return regularization


def update_mu(Z, W, Xi_, nu, gamma, min_margin):
    """
    Update mu by combining numerator (X_i.) and denominator
    """
    den_mu = get_denominator_mu(Z, W, nu, gamma)
    mu = Xi_ / den_mu
    np.clip(mu, min_margin, None, mu)
    return mu


def update_nu(Z, W, X_j, mu, gamma, min_margin):
    """
    Update nu by combining numerator (X_.j) and denominator
    """
    den_nu = get_denominator_nu(Z, W, mu, gamma)
    nu = X_j / den_nu
    np.clip(nu, min_margin, None, nu)
    return nu


def _compute_static_mixture_part_Lc(
        regularization_mode, regularize_row, regularize_col, P_r, P_c,
        Z_, W_, log_alpha, log_beta
):
    """
    Computes the terms in alpha and beta in complete data log likelihood
    """
    # the terms in alpha and beta in Lc
    if (not (regularize_row or regularize_col) or regularization_mode == 'all'):
        Lc_part = Z_.dot(log_alpha) + W_.dot(log_beta)
    elif regularization_mode == 'mixture':
        not_P_r = ~ P_r
        not_P_c = ~ P_c
        Lc_part = ((not_P_r[:, nax] * log_alpha[nax, :]).sum() +
                   (not_P_c[:, nax] * log_beta[nax, :]).sum())
    return Lc_part


def compute_Lc_static(
        model_type, estimated_margins,
        regularization_mode, regularize_row, regularize_col,
        P_r, P_c, log_alpha, log_beta, gamma, mu, nu, Xi_, X_j,
        Z_, W_, X_kl, min_float
):
    """
    computes the complete data log likelihood for a static HLBM (T = 1)
    """
    # the terms in alpha and beta in Lc
    Lc_mixture = _compute_static_mixture_part_Lc(
        regularization_mode, regularize_row, regularize_col,
        P_r, P_c, Z_, W_, log_alpha, log_beta
    )
    # the terms in gamma (and mu, nu) in Lc
    Lc_density = compute_Lc_static_density(
        model_type, estimated_margins,
        gamma, mu, nu, Xi_, X_j,
        X_kl, min_float
    )
    return Lc_mixture + Lc_density


def entropy_static(ZW, min_float):
    """
    Entropy, static part
    """
    # ZW[absent_nodes] = 0, so no pb
    return - (ZW * np.log(ZW + min_float)).sum()


def entropy_appearing(ZW, appearing_nodes, min_float):
    """
    Entropy, appearing nodes part (if any)
    """
    entr = 0.
    for t, _ind_app_t in appearing_nodes.items():
        entr -= (ZW[t] * np.log(ZW[t] + min_float)).sum(0, where=appearing_nodes).sum()
    return entr


def entropy_present(ZW, qzw, min_float):
    """
    Entropy, present nodes part
    """
    # ZW[absent_nodes] = 0, so no pb
    return (ZW[:-1, :, :, nax] * qzw[:1] * np.log(qzw[:1] + min_float)).sum()


def entropy_dynamic(
    ZW,
    qzw,
    appearing_nodes,
    min_float
):
    """
    Total entropy
    """
    H = 0.
    H += entropy_static(ZW, min_float)
    if appearing_nodes is not None:
        H += entropy_appearing(ZW, appearing_nodes, min_float)
    H += entropy_present(ZW, qzw, min_float)
    return H
