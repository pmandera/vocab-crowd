#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import numpy as np
import pandas as pd


def flatten(d, parent_key='', sep='_'):
    """
    Flatten distionary concatenating keys.

    Taken from: http://stackoverflow.com/a/6027615
    """
    items = []
    for k, v in list(d.items()):
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(list(flatten(v, new_key, sep=sep).items()))
        else:
            items.append((new_key, v))
    return dict(items)


def mds(space, words, method=None, metric=None):
    if method is None:
        import sklearn.decomposition as sd
        method = sd.PCA(n_components=2)

    if metric is not None:
        m = space.matrix_distances(words, words, metric=metric)
    else:
        m = space.word_vectors_matrix(words)

    index = pd.Index(words, name='words')

    return pd.DataFrame(method.fit_transform(m), index=index)


def plot_2d(df, fout, colors=None, annotate=True,
            size=2, marker='o', fontsize=5, figsize=(10, 10)):
    """Plot scatterplot."""
    import matplotlib.pyplot as plt
    labs = list(df.index)
    m = df.values

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    if colors is not None:
        ax.scatter(m[:, 0], m[:, 1], c=colors, s=size, marker=marker)
    else:
        ax.scatter(m[:, 0], m[:, 1], s=size, marker=marker)

    if annotate:
        for nr, w in enumerate(labs):
            x, y = m[nr, 0], m[nr, 1]
            ax.annotate(w, (x, y), fontsize=fontsize)

    fig.savefig(fout)


def se(x):
    std = np.std(x)
    n = len(x)
    return std/np.sqrt(n)
