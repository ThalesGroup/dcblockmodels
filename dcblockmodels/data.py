"""Methods to sample data from static or dynamic LBM/SBM"""

import numpy as np


def sample_edge(mu_it, nu_it, gamma_tkl, block_sparsity_matrix_tkl):
    """
    Sample a given edge with paramters mu_it, nu_it, gamma_tkl
    in a Poisson distribution, while keeping a sparsity of block_sparsity_matrix_tkl
    (defined in [0, 1))
    """
    if np.random.rand() < block_sparsity_matrix_tkl:
        return 0
    return np.random.poisson(lam=mu_it * nu_it * gamma_tkl)


def generate_data(
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
        noise_level,
        with_absent_nodes,
        absent_nodes,
        dtype,
        block_sparsity_matrix):
    """
    noise_level is the std of gaussian noise
    block_sparsity_matrix is 1 - connectivity_matrix
    in the binary case
    """
    assert model_type in ['SBM', 'LBM']
    assert with_margins in [False, True]
    assert dtype in ['int32', 'int64']

    # parses the inputs
    N = dimensions['N']
    Kz = n_clusters['Kz']
    alpha = prior_init['alpha']
    if model_type == 'LBM':
        beta = prior_init['beta']
        D = dimensions['D']
        Kw = n_clusters['Kw']
    elif model_type == 'SBM':
        D = N
        Kw = Kz
        W = None  # for consistency

    print('Data characteristics: ')
    print(f'   - T, N, D = {T, N, D}')
    if T == 1:
        print('   - Static ' + model_type)
    else:
        print('   - Dynamic ' + model_type)

    if T > 1:
        pi = prior_trans['pi']
        if model_type == 'LBM':
            rho = prior_trans['rho']

    if not directed:
        assert model_type == 'SBM'
        print('   - Undirected')
    else:
        if model_type == 'SBM':
            print('   - Directed')

    if not self_loops:
        assert model_type == 'SBM' or N == D
        print('   - Without self-loops')
    else:
        print('   - With self-loops')

    if block_sparsity_matrix is not None:
        assert block_sparsity_matrix.shape == gamma.shape
    else:
        block_sparsity_matrix = np.zeros_like(gamma, dtype=dtype)

    # the margins are either generated in generate_margins()
    # or we generate constant margins equal to 1
    if with_margins:
        print('   - With margins')
        mu = margins['mu']
        nu = margins['nu']
    else:
        print('   - Without margins')
        if T > 1:
            mu = np.ones((T, N), dtype=dtype)
            if directed:
                nu = np.ones((T, D), dtype=dtype)
        else:
            mu = np.ones((N), dtype=dtype)
            if directed:
                nu = np.ones((D), dtype=dtype)

    if with_absent_nodes and T > 1:
        absent_row_nodes = absent_nodes['absent_row_nodes']
        res_row = {t: 0 for t in range(T)}
        for t, i in absent_row_nodes:
            res_row[t] += 1
        print(f'   - absent row nodes {res_row}')
        if directed:
            absent_col_nodes = absent_nodes['absent_col_nodes']
            res_col = {t: 0 for t in range(T)}
            for t, i in absent_col_nodes:
                res_col[t] += 1
            print(f'   - absent col nodes {res_col}')

    # converts the static parameters to
    # dynamic ones for genericity
    if gamma.ndim == 2 and T > 1:
        gamma = np.stack([gamma for _ in range(T)])

    if mu.ndim == 1 and T > 1:
        mu = np.stack([mu for _ in range(T)])

    if directed:
        if nu.ndim == 1 and T > 1:
            nu = np.stack([nu for _ in range(T)])

    if block_sparsity_matrix.ndim == 2 and T > 1:
        block_sparsity_matrix = np.stack([block_sparsity_matrix for _ in range(T)])

    # generates the latent processes
    Z0 = np.random.choice(Kz, size=(N), p=alpha)

    if model_type == 'LBM':
        W0 = np.random.choice(Kw, size=(D), p=beta)

    if T > 1:
        Z = np.zeros((T, N), dtype=dtype)
        Z[0] = Z0.copy()
        for t in range(1, T):
            for i in range(N):
                k = Z[t - 1, i]
                Z[t, i] = np.random.choice(Kz, p=pi[k, :])

        if model_type == 'LBM':
            W = np.zeros((T, D), dtype=dtype)
            W[0] = W0.copy()
            for t in range(1, T):
                for j in range(D):
                    k = W[t - 1, j]
                    W[t, j] = np.random.choice(Kw, p=rho[k, :])
    else:
        Z = Z0
        if model_type == 'LBM':
            W = W0

    # Generate the data matrix with the latent processes
    if T > 1:  # pylint: disable=R1702
        X = np.zeros((T, N, D), dtype=dtype)
        if not directed:
            # we have model_type == SBM
            for t in range(T):
                for i in range(N):
                    for j in range(i + 1):
                        k, l = Z[t, i], Z[t, j]
                        val = sample_edge(
                            mu[t, i], mu[t, j],
                            gamma[t, k, l],
                            block_sparsity_matrix[t, k, l]
                        )
                        X[t, i, j] = val
                        X[t, j, i] = val
        else:
            for t in range(T):
                for i in range(N):
                    for j in range(D):
                        if model_type == 'LBM':
                            k, l = Z[t, i], W[t, j]
                        else:
                            k, l = Z[t, i], Z[t, j]
                        val = sample_edge(
                            mu[t, i], nu[t, j],
                            gamma[t, k, l],
                            block_sparsity_matrix[t, k, l]
                        )
                        X[t, i, j] = val
    else:
        X = np.zeros((N, D), dtype=dtype)
        if not directed:
            for i in range(N):
                for j in range(i + 1):
                    k, l = Z[i], Z[j]
                    val = sample_edge(
                        mu[i], mu[j],
                        gamma[k, l],
                        block_sparsity_matrix[k, l]
                    )
                    X[i, j] = val
                    X[j, i] = val
        else:
            for i in range(N):
                for j in range(D):
                    if model_type == 'LBM':
                        k, l = Z[i], W[j]
                    else:
                        k, l = Z[i], Z[j]
                    val = sample_edge(
                        mu[i], nu[j],
                        gamma[k, l],
                        block_sparsity_matrix[k, l]
                    )
                    X[i, j] = val

    if not self_loops:
        if T > 1:
            for t in range(T):
                np.fill_diagonal(X[t], 0)
        else:
            np.fill_diagonal(X, 0)

    # adds noise to the resulting matrix
    noise = np.random.normal(loc=0., scale=noise_level, size=X.shape).astype(dtype)

    if not self_loops:
        if T == 1:
            np.fill_diagonal(noise, 0)
        else:
            for t in range(T):
                np.fill_diagonal(noise[t], 0)

    np.add(X, noise, out=X)
    np.clip(X, a_min=0, a_max=None, out=X)

    if with_absent_nodes and T > 1:
        for t, i in absent_row_nodes:
            X[t, i, :] = 0
            if not (directed and model_type == 'SBM'):
                Z[t, i] = -1
        if directed:
            for t, j in absent_col_nodes:
                X[t, :, j] = 0
                if model_type == 'SBM':
                    # see below
                    pass
                elif model_type == 'LBM':
                    W[t, j] = -1

            # in the directed case with SBM and different row and col absent nodes a node is
            # considered absent in Z if it is absent in the row AND the col
            if model_type == 'SBM':
                for t, i in absent_row_nodes:
                    if (t, i) in absent_col_nodes:
                        Z[t, i] = -1

    if T == 1:
        sparsity = (X == 0).sum() / (N * D)
        print(f'   - Sparsity {100 * sparsity: .2f}%')
    else:
        print('   - Sparsity', end=' ')
        for t in range(T):
            sparsity = (X[t] == 0).sum() / (N * D)
            print(f't = {t} {100 * sparsity: .2f}%', end='|')

    if not with_absent_nodes:
        dynamic_offset = int(T > 1)
        if (X.sum(dynamic_offset) == 0).any() or (X.sum(dynamic_offset + 1) == 0).any():
            print(
                'Warning : with_absent_nodes is False but there are '
                'nodes with a zero in- or out-degree. They will be '
                'classified to cluster -1, which is different from'
                'their true cluster. The model parameters are probably '
                'too low'
            )
    return X, Z, W


def AR1_process_margins(x0, a, sigma2, T):
    """
    given x0, array of margins at time t=0, generates
    margins with T samples for each margin
    x[t+1] = N(a * x[t] + c, sigma2)
    where c is adapted so that the margins are
    increasing over time
    """
    max_trials = 5000
    n_trials = 0

    def sample(x0, a, sigma2, T):
        c = .1 + x0 * (1. - a)
        # c = 0.
        res = np.zeros(shape=(T, x0.shape[0]), dtype='float32')
        res[0] = x0
        for t in range(1, T):
            mu = a * res[t - 1] + c
            cov = sigma2 * np.eye(x0.shape[0])
            res[t] = np.random.multivariate_normal(mu, cov, 1)[0]
        return res

    res = np.full((T, x0.shape[0]), -1.)
    while (res <= 0.).any():
        res = sample(x0, a, sigma2, T)
        n_trials += 1
        if n_trials == max_trials:
            raise Exception('Could not sample all non negative values from AR1(x0, a, sigma2)')
    return res


def generate_margins(
        T, N, D, constant_margins, start, stop, step,
        directed, order_power_law,
        ar_margins=None, a_ar=None, sigma2_ar=None
):
    """
    Generates margins : arrays of shape (T, N) or (N)
    values x sampled in np.arange(start, stop, step)
    with probability propto x ^ order_power_law
    """
    assert order_power_law < 0
    if D is None:
        D = N
    nax = np.newaxis

    if constant_margins:
        print('Constant margins')
    else:
        print('Variable margins')

    values_margins = np.arange(start, stop, step)
    proba_distrib = values_margins ** order_power_law
    np.divide(proba_distrib, proba_distrib.sum(), out=proba_distrib)
    if T == 1 or constant_margins or ar_margins:
        mu = np.random.choice(
            a=values_margins,
            size=(N),
            p=proba_distrib
        )
        if directed:
            nu = np.random.choice(
                a=values_margins,
                size=(D),
                p=proba_distrib
            )
    else:
        mu = np.random.choice(
            a=values_margins,
            size=(T, N),
            p=proba_distrib
        )
        if directed:
            nu = np.random.choice(
                a=values_margins,
                size=(T, D),
                p=proba_distrib
            )
    if T > 1 and constant_margins:
        mu = np.concatenate([mu[nax, :] for t in range(T)], axis=0)
        if directed:
            nu = np.concatenate([nu[nax, :] for t in range(T)], axis=0)

    if T > 1 and ar_margins:
        mu = AR1_process_margins(mu, a_ar, sigma2_ar, T)
        if directed:
            nu = AR1_process_margins(nu, a_ar, sigma2_ar, T)

    if directed:
        return mu, nu
    return mu, None


def generate_diag_transition_matrix(Kzw, val_diag):
    """
    val_diag : the proba of transition to the same state
    """
    val_off_diag = (1. - val_diag) / (Kzw - 1)
    return (val_diag - val_off_diag) * np.eye((Kzw)) + val_off_diag


def generate_initial_proportions(Kzw, alpha_dirichlet):
    """
    generates a np.array of shape (Kzw, ) from a dirichlet
    distribution with constant parameter alpha_dirichlet
    """
    return np.random.dirichlet(np.full(Kzw, alpha_dirichlet), size=1)[0]


def sample_absent_nodes(T, ND, min_proba_t=None, max_proba_t=None, proba_absent=None):
    """
    Samples the time and node index of absent nodes.
    If min_proba_t and max_proba_t are given, samples for each
    time step t a probability p_t from Unif(min_proba_t, max_proba_t),
    then samples the number n_abs_t of absent nodes at time t
    from Binom(ND, p_t), then samples n_abs_t nodes without
    replacement.
    If min_proba_t=None, max_proba_t=None and proba_absent is not
    None, then all p_t = proba_absent
    """
    if min_proba_t is not None and max_proba_t is not None:
        assert proba_absent is None
        proba_absent = np.random.uniform(low=min_proba_t, high=max_proba_t, size=T)

    absent = []
    n_absent_t = np.random.binomial(ND, proba_absent, size=(T))
    for t, n_abs in enumerate(n_absent_t):
        i_absent_t = np.sort(np.random.choice(ND, size=n_abs, replace=False))
        absent += list(zip([t] * n_abs, list(i_absent_t)))

    return absent
