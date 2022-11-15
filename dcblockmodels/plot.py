"""Plotting functions for model results"""

import time
import warnings

import numpy as np
import pandas as pd
import networkx as nx

import prince

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


def plot_criterions(model, thr_decrease, i_start=0, i_end=-1, legend=True):
    """
    Plots the criterions as a function of the iteration
    number for each initialization of a model.

    In the case of (d)LBM models, two criteria are
    computed for a given E step. We thus plot the
    2 * n_iter_tot values of this criterion.

    The iterations steps where there is a "decrease" in
    the criterion are highlighed with a 'o' marker.
    thr_decrease determines by how much an iteration
    must increase the likelihood to be considered "decreasing"
    """
    assert thr_decrease > 0
    _f, ax = plt.subplots(figsize=(16, 5))
    pal = sns.color_palette('colorblind', n_colors=model.n_init)

    for init in range(model.n_init):
        crits = model.all_iter_criterions[init]
        if hasattr(model, 'all_intermediate_iter_criterions'):
            interm_crits = model.all_intermediate_iter_criterions[init]
            # all_crits = [interm[0], crit[0], interm[1], crit[1],...]
            all_crits = np.array([j
                                  for i in zip(interm_crits, crits)
                                  for j in i])
        else:
            all_crits = np.array(crits)
        all_crits = all_crits[i_start: i_end]

        diff = all_crits[1:] - all_crits[:-1]
        increase = (diff > -thr_decrease)
        if not increase.all():
            iters_pb = np.where(~increase)[0] + 1
            ax.plot(iters_pb, all_crits[iters_pb], marker='o', color=pal[init], lw=0)

        ax.plot(all_crits, label=f'{init}', alpha=.9, color=pal[init], lw=2.)
    if legend:
        ax.legend(loc='best', title='initializations', fancybox=True)
    return ax


def CA_plot(X, Z, W, absent_row_nodes=None, absent_col_nodes=None, ax=None):
    """
    plots the projction of the rows and the columns of
    the matrix X onto the factorial plane found by
    correspondance analysis.
    """
    warnings.filterwarnings("ignore", category=FutureWarning)

    assert X.ndim == 2

    N, D = X.shape
    present_row = np.setdiff1d(np.arange(N), absent_row_nodes)
    present_col = np.setdiff1d(np.arange(D), absent_col_nodes)
    X_ = X[np.ix_(present_row, present_col)]
    X_ = X_ + 1e-10
    N_, D_ = X_.shape

    if N == D and not np.array_equal(present_row, present_col):
        print('Special case: SBM with different absent row and col nodes.')

    # for directed SBM with different absent
    # row and col nodes, X_ is not square
    # and W_ is taken from Z with absent col nodes
    Z_ = Z[present_row]
    if W is not None:
        W_ = W[present_col]
    else:
        W_ = Z[present_col]

    if ax is None:
        _f, ax = plt.subplots(1, 2, figsize=(10, 5))
    row_clusters = np.unique(Z_)
    col_clusters = np.unique(W_)
    n_clusters_tot = row_clusters.shape[0] + col_clusters.shape[0]

    ca = prince.CA(
        n_components=10,
        n_iter=30,
        copy=True,
        check_input=True,
        engine='auto',
        random_state=42
    )

    df = pd.DataFrame(X_)
    ca = ca.fit(df)
    row_factor_score = ca.row_coordinates(df).values
    col_factor_score = ca.column_coordinates(df).values
    pal = sns.color_palette('colorblind', n_colors=n_clusters_tot)
    pal_rows = pal[:row_clusters.shape[0]]
    pal_cols = pal[row_clusters.shape[0]:]

    for k in row_clusters:
        ix = np.where(Z_ == k)[0]
        prop = 100 * ix.shape[0] / N_
        ax[0].scatter(row_factor_score[ix, 0],
                      row_factor_score[ix, 1],
                      c=[pal_rows[k]],
                      label=f'{k}: {prop:.0f}%',
                      edgecolors='black',
                      alpha=.8, s=200)
    for l in col_clusters:
        ix = np.where(W_ == l)[0]
        prop = 100 * ix.shape[0] / D_
        ax[1].scatter(col_factor_score[ix, 0],
                      col_factor_score[ix, 1],
                      c=[pal_cols[l]],
                      label=f'{l}: {prop:.0f}%',
                      edgecolors='black',
                      alpha=.8, s=200)

    exp_var = 100 * np.array(ca.explained_inertia_)
    s = 'explained inertia {:.2f}%'
    for i in range(2):
        ax[i].set_xlabel(s.format(exp_var[0]))
        ax[i].set_ylabel(s.format(exp_var[1]))

    for x, y, name in zip(row_factor_score[:, 0],
                          row_factor_score[:, 1],
                          df.index.values):
        ax[0].text(x, y, name)
    for x, y, name in zip(col_factor_score[:, 0],
                          col_factor_score[:, 1],
                          df.columns.values):
        ax[1].text(x, y, name)

    ax[0].legend(title='clusters')
    ax[1].legend(title='clusters')
    warnings.filterwarnings('default', category=FutureWarning)
    return ax


def plot_reorganised_matrix(X, Z, logscale=False, light=1.3, snapshot_titles=None):
    """
    Reorganized matrix with line/column permutations
    """
    T = X.shape[0]
    _f, ax = plt.subplots(T, 1, figsize=(6, 6 * T))
    cmap = sns.cubehelix_palette(light=light, as_cmap=True)
    for t in range(T):
        if snapshot_titles is not None:
            ax[t].title.set_text(snapshot_titles[t])
        indices = np.argsort(Z[t].astype(int))
        X_reorg = X[t, indices, :]
        X_reorg = X_reorg[:, indices]
        df_plot = pd.DataFrame(X_reorg)
        if logscale:
            df_plot = np.log10(df_plot + 1)  # pour logscale
        sns.heatmap(df_plot, ax=ax[t], cmap=cmap, linewidths=.005)
        # plots the lines that separates the blocks
        _, unique_indices = np.unique(Z[t], return_counts=True)
        x_indices = np.cumsum(unique_indices)
        for x in x_indices:
            ax[t].axvline(x, linewidth=2.5)
            ax[t].axhline(x, linewidth=2.5)


def plot_connectivity_matrix(gamma, subclusters=None):
    """
    Row and column normalized connectivity matrix
    at a given time step (or constant connectivity)
    """
    if subclusters is not None:
        gamma = gamma[np.ix_(subclusters, subclusters)]

    Kz, Kw = gamma.shape
    _f, ax = plt.subplots(1, 2, figsize=(2 * 6, 5))
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)

    g1 = gamma / gamma.sum(axis=1, keepdims=True)
    g2 = gamma / gamma.sum(axis=0, keepdims=True)

    sns.heatmap(g1, cmap=cmap, ax=ax[0], annot=True, fmt='.2f')
    sns.heatmap(g2, cmap=cmap, ax=ax[1], annot=True, fmt='.2f')

    for y in range(Kz + 2):
        ax[0].axhline(y, linewidth=2.5)
    for x in range(Kw + 2):
        ax[1].axvline(x, linewidth=2.5)

    if subclusters is not None:
        for i in range(2):
            ax[i].set_xticklabels(subclusters)
            ax[i].set_yticklabels(subclusters)

    for i in range(2):
        ax[i].set_xlabel('destination cluster')
        ax[i].set_ylabel('origin cluster')
        ax[i].set_title('destination connection profiles in %')


def to_gephi(X, Z=None, filename=None, model=None):
    """
    Plot the matrix in the Gephi (Open Graph Viz Platform) format
    """
    assert X.ndim == 2

    if not filename:
        filename = f'graph_{time.time()}'

    G = nx.from_numpy_matrix(np.matrix(X, dtype=[('weight', int)]))
    deg = X.sum(axis=0)
    for i in G.nodes:
        G.node[i]['node_degree'] = deg[i]

    if Z is not None:
        assert Z.ndim == 1
        for i in G.nodes:
            G.node[i]['true_cluster'] = str(int(Z[i]))
    if model:
        for i in G.nodes:
            G.node[i]['found_cluster'] = str(int(model.best_Z[0, i]))

    nx.write_gexf(G, filename + '.gexf')


def plot_alluvial(Z, df_stations=None, dates=None, dataset='mtr'):
    """
    Plot the clusters in an alluvial graph (only lines or columns at a time)
    """

    def parse_stations_bart(stat_numbers):
        stat_numbers = np.array(list(stat_numbers)).astype('int')
        quadrigrams = df_stations.iloc[stat_numbers]['quadrigram'].values
        res = '<em>Stations :</em><br> ' + "<br> ".join(quadrigrams) + "<br>"
        return res

    def parse_stations_mtr(stat_numbers):
        """
        from a set of stations to an htlm
        formated string that describes the nodes
        transitionning between two clusters
        """
        stat_numbers = np.array(list(stat_numbers)).astype('int')
        d = pd.DataFrame({'station_number': stat_numbers})

        d = df_stations.merge(
            d,
            on='station_number',
            how='inner'
        ).groupby('line_').agg({'station_code': list}).reset_index().values

        res = '<em>Stations :</em><br>'
        for l in range(d.shape[0]):
            res += '<b>' + d[l, 0] + '</b>' + '<br>'
            trigrams = np.unique(d[l, 1])
            for i, x in enumerate(trigrams):
                if (i > 0) and (i % 5 == 0):
                    res += '<br>'
                res += ' ' + x
            res += '<br>'
        return res

    if dates is not None:
        assert len(dates) == Z.shape[0]

    T = Z.shape[0]
    K_values = np.sort(np.unique(Z.ravel()).astype(int))
    K = K_values.shape[0]

    trans = [set(np.where(np.logical_and(Z[t] == s, Z[t + 1] == d))[0])
             for t in range(T - 1)
             for s in K_values
             for d in K_values]

    # colors_plotly = ['red', 'green', 'blue', 'yellow', 'orange']
    colors_plotly = np.array(['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure',
                              'beige', 'bisque', 'black', 'blanchedalmond', 'blue',
                              'blueviolet', 'brown', 'burlywood', 'cadetblue',
                              'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
                              'cornsilk', 'crimson', 'cyan', 'darkcyan',
                              'darkmagenta', 'darkolivegreen', 'darkorange',
                              'darkorchid', 'darkred', 'darksalmon', 'darkseagreen'])

    # Nodes
    if dates is not None:
        if dataset == 'mtr':
            label = [f"K{k}, {dates[t]}"
                     for t in range(T)
                     for k in K_values]
        elif dataset == 'bart':
            label = [f"K{k} {dates[t][1]}"
                     for t in range(T)
                     for k in K_values]
    else:
        label = [f"K{k} t{t}"
                 for t in range(T)
                 for k in K_values]
    node_color = np.random.choice(colors_plotly, K, replace=False).tolist() * T

    # Links
    source = [s + t * K
              for t in range(T - 1)
              for s in range(K)
              for d in range(K)]
    target = [d + (t + 1) * K
              for t in range(T - 1)
              for s in range(K)
              for d in range(K)]

    value = np.zeros((K * K * (T - 1)))
    for t in range(T - 1):
        for s in range(K):
            for d in range(K):
                flow_tsd = len(trans[t * K**2 + s * K + d])
                value[t * K**2 + s * K + d] = flow_tsd if flow_tsd > 0 else -1
    value = value.astype(int)

    link_colors = ['grey' if s == d else 'black'
                   for t in range(T - 1)
                   for s in range(K)
                   for d in range(K)]

    if df_stations is not None:
        if dataset == 'mtr':
            link_label = [parse_stations_mtr(x)
                          if len(x) > 0 else 'None'
                          for x in trans]
        elif dataset == 'bart':
            link_label = [parse_stations_bart(x)
                          if len(x) > 0 else 'None'
                          for x in trans]
    else:
        link_label = [str(x) if len(x) > 0 else 'None' for x in trans]

    if dates is not None:
        if dataset == 'mtr':
            title = ('MTR Network sum-up <br> Station groups evolution '
                     f'from {dates[0]} to {dates[-1]}'
                     )
        elif dataset == 'bart':
            title = ('BART Network sum-up <br> Station'
                     ' groups evolution '
                     f'from {dates[0][0]} {dates[0][1]}h '
                     f'to {dates[-1][0]} {dates[-1][1]}h'
                     )
    else:
        title = f'Clusters evolution over {T} snapshots'

    # Layout
    layout = dict(title=title, font=dict(size=12), height=700, width=1200)

    # Figure
    f = go.Figure(data=[go.Sankey(

        valueformat=".f",
        valuesuffix=" stations",

        node=dict(
            pad=30,
            thickness=10,
            line=dict(color="black", width=0.5),
            label=label,
            color=node_color,
            hoverlabel=dict(
                bgcolor=node_color,
                font=dict(
                    size=15
                )
            )
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            label=link_label,
            color=link_colors,
            hoverlabel=dict(
                bgcolor='white',
                font=dict(
                    size=12
                )
            )
        ),
        textfont=dict(
            size=7,
            color='black'
        )

    )],
        layout=layout
    )

    return f


def plot_gamma(gamma, dates=None, step=None, show_cluster_ids=True):
    """
    Plot the gamma matrix in Matplotlib
    """
    assert gamma.ndim == 3
    if dates is not None:
        assert gamma.shape[0] == len(dates)
    if step is None:
        step = 1

    K = gamma.shape[1]
    T = gamma.shape[0]
    time_range = np.arange(0, T, step)
    dates = [d for i, d in enumerate(dates) if i % step == 0]
    _f, ax = plt.subplots(K, K, figsize=(5 * K, 5 * K), sharex=True, sharey=True)

    for k in range(K):
        if show_cluster_ids:
            ax[k, 0].set_ylabel(f'from cluster {k}')
            ax[K - 1, k].set_xlabel(f'to cluster {k}')
            ax[0, k].set_xlabel(f'to cluster {k}')
            ax[0, k].xaxis.set_label_position('top')
        for l in range(K):
            ax[k, l].plot(gamma[:, k, l])
            if dates is not None:
                ax[k, l].set_xticks(time_range)
                ax[k, l].set_xticklabels(dates)


def plot_margins(model, always_present, dates=None, subset=None):
    """
    always_present = utils.get_always_present_nodes()
    subset : subset of nodes to be considered
    dates : ndarray of strings of dates
    """
    _f, ax = plt.subplots(2, 1, figsize=(16, 12))

    if subset is not None:
        always_present = np.intersect1d(subset, always_present)

    ax[0].plot(model.best_mu[:, always_present])
    ax[1].plot(model.best_nu[:, always_present])

    if dates is not None:
        ax[0].set_xticklabels(dates)
        ax[1].set_xticklabels(dates)


def plot_mu_nu_during_optim(debug_values_mu_nu, indexes=None):
    """
    Plots mu_{i}^t^(c) as a function of (c),
    for each timestep t, each i in indexes and each run
    """
    n_init = len(debug_values_mu_nu)
    T = debug_values_mu_nu[0][0].shape[0]
    _f, ax = plt.subplots(n_init, 1, figsize=(8, 4 * n_init))

    if indexes is None:
        indexes = range(debug_values_mu_nu[0][0].shape[1])
    for init in range(n_init):
        mu_nu = np.array(debug_values_mu_nu[init])
        mu_nu = mu_nu[:, :, ] if indexes is not None else mu_nu
        for t in range(T):
            for i in indexes:
                ax[init].plot(
                    mu_nu[:, t, i],
                    label=f'mu/nu_t={t},i={i}'
                )
        ax[init].legend()
        ax[init].title.set_text(f'Run {init + 1}')
    plt.tight_layout()


def plot_alphas_during_optim(debug_values_alpha_beta):
    """
    Plots \alpha_{k}^(c) as a function of (c),
    for each cluster and run
    """
    n_init = len(debug_values_alpha_beta)
    Kzw = debug_values_alpha_beta[0][0].shape[0]
    _f, ax = plt.subplots(n_init, 1, figsize=(8, 4 * n_init))

    for init in range(n_init):
        alpha_beta = np.exp(debug_values_alpha_beta[init])
        for k in range(Kzw):
            ax[init].plot(
                alpha_beta[:, k],
                label=f'alpha/beta_{k}'
            )
        ax[init].legend()
        ax[init].title.set_text(f'Run {init + 1}')
    plt.tight_layout()


def plot_pi_rho_during_optim(debug_values_pi_rho):
    """
    Plots \alpha_{k}^(c) as a function of (c),
    for each cluster and run
    """
    n_init = len(debug_values_pi_rho)
    Kzw = debug_values_pi_rho[0][0].shape[0]
    _f, ax = plt.subplots(n_init, 1, figsize=(8, 4 * n_init))

    for init in range(n_init):
        pi_rho = np.exp(debug_values_pi_rho[init])
        for k in range(Kzw):
            for l in range(Kzw):
                ax[init].plot(
                    pi_rho[:, k, l],
                    label=f'pi_rho_{(k, l)}'
                )
        ax[init].legend()
        ax[init].title.set_text(f'Run {init + 1}')
    plt.tight_layout()


def plot_gamma_during_optim(debug_values_gamma):
    """
    Plots gamma_{kl}^(c) as a function of (c),
    for each pair of cluster k and l and run
    """
    assert debug_values_gamma[0][0].ndim == 2

    n_init = len(debug_values_gamma)
    Kzw = debug_values_gamma[0][0].shape[0]
    _f, ax = plt.subplots(n_init, 1, figsize=(8, 4 * n_init))

    for init in range(n_init):
        gamma = np.exp(debug_values_gamma[init])
        for k in range(Kzw):
            for l in range(Kzw):
                ax[init].plot(
                    gamma[:, k, l],
                    label=f'gamma_{(k, l)}'
                )
        ax[init].legend()
        ax[init].title.set_text(f'Run {init + 1}')
    plt.tight_layout()
