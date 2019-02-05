#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import statsmodels.stats.stattools as st


def spelling_by_profile(vt, s1, s2, ld_query):
    alias1, q1 = s1
    alias2, q2 = s2

    vt1 = vt.query_by_profile(q1)
    vt2 = vt.query_by_profile(q2)

    if ld_query:
        vt1_ld = vt1.ld.query(ld_query)
        vt2_ld = vt2.ld.query(ld_query)
    else:
        vt1_ld = vt1.ld
        vt2_ld = vt2.ld

    vt1_acc = vt1_ld.groupby(
        ('lexicality', 'spelling'))['accuracy'].aggregate(
            {'accuracy_mean': np.mean, 'nobs': lambda x: x.shape[0]})
    vt2_acc = vt2_ld.groupby(('lexicality', 'spelling'))['accuracy'].aggregate(
            {'accuracy_mean': np.mean, 'nobs': lambda x: x.shape[0]})

    accuracy = vt1_acc.join(vt2_acc, how='outer',
                            lsuffix='_' + alias1, rsuffix='_' + alias2)

    vt1_rt = vt1_ld.query('accuracy == 1').groupby(
        ('lexicality', 'spelling'))[['rt_zscore', 'rt']].mean()
    vt2_rt = vt2_ld.query('accuracy == 1').groupby(
        ('lexicality', 'spelling'))[['rt_zscore', 'rt']].mean()

    rt = vt1_rt.join(vt2_rt, how='outer',
                     lsuffix='_' + alias1, rsuffix='_' + alias2)

    return accuracy.join(rt, how='outer')


def adjboxInterval(v, a=-4, b=3, coef=1.5):
    """Return an interval for adjusted boxplot.

    The adjusted boxplot marks the observations that fall outside the inteval
    defined by Hubert & Vandervieren (2008).

    References:
    Hubert, M., & Vandervieren, E. (2008). An adjusted boxplot for skewed
    distributions. Computational Statistics & Data Analysis, 52(12),
    5186–5201. http://doi.org/10.1016/j.csda.2007.11.008
    """
    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    mc = st.medcouple(v)
    if mc > 0:
        return (q1 - coef * iqr * np.exp(a * mc),
                q3 + coef * iqr * np.exp(b * mc))
    else:
        return (q1 - coef * iqr * np.exp(-b * mc),
                q3 + coef * iqr * np.exp(-a * mc))
