"""
##############################################################
                            functions
##############################################################
"""


# calculate gini (second way)
def calc_gini_2(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]

    # sort rows by prediction column
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:, 0].argsort()][::-1, 0]
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]

    # get lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(1 / n_samples, 1, n_samples)

    # get gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # normalize by true gini coefficient
    return G_pred / G_true


# calculate r2 for one-factor model
def _calc_r2(x, y, b):
    bins = [a for a in zip(b._gaps, b._woes)]
    data = pd.concat([x.rename('x'), y.rename('y')], axis=1)

    # transform according to binning
    # nans
    data.loc[pd.isnull(data['x']), 'x_woe'] = bins[bins[0][0] is None][1]

    # non-nans
    for bi in [bb for bb in bins if bb[0][0] is not None]:
        low = bi[0][0]
        high = bi[0][1]
        data.loc[(data['x'] >= low) & (data['x'] < high), 'x_woe'] = bi[1]

    # fit linear regression
    lin = LinearRegression(fit_intercept=True)
    lin.fit(data['x_woe'][:, None], y)
    r2 = lin.score(data['x_woe'][:, None], y)
    return r2


# remove values that occur more than once
def _delete_duplicates(arr):
    new_arr = list()
    for a in arr:
        if a in new_arr:
            new_arr.remove(a)
        else:
            new_arr.append(a)
    return new_arr


# calculate woe
def _calc_WoE(gap_bads, bads, gap_goods, goods):
    bads_share = (len(gap_bads) + 0.5) / (len(bads) + 0.5) * 1.0
    goods_share = (len(gap_goods) + 0.5) / (len(goods) + 0.5) * 1.0
    val = goods_share / bads_share
    return math.log(val), goods_share, bads_share


# herfindahl–hirschman index
def _calc_HHI(gaps_counts):
    total = 0
    hh_shares = []
    for counts in gaps_counts:
        total += counts[0] + counts[1]
    for counts in gaps_counts:
        hh_shares.append((counts[0] + counts[1]) / total)
    hhi = np.sum([hh ** 2 for hh in hh_shares])
    return hhi


# check if values are monotonic
def _is_monotonik(vals):
    # compute differences between woe values
    difs = [j - i for i, j in zip(vals[:-1], vals[1:])]
    if len(difs) == 0:
        return True

    # check that the sign of differences does not change
    flag = difs[0] < 0
    for dif in difs:
        if (dif < 0) != flag:
            return False
    return True


# calculate gini using sas formula
def _calc_gini(gaps_counts):
    # sort by descending number of goods in the gap
    s_gaps_counts = sorted(gaps_counts, key=lambda x: x[0], reverse=True)

    # first term of numerator
    first_sl = 0
    for i in range(1, len(s_gaps_counts)):
        first_sl += s_gaps_counts[i][0] * np.sum([gc[1] for gc in s_gaps_counts[:i]])

    # second term of numerator
    second_sl = 0
    for gap_counts in s_gaps_counts:
        second_sl += gap_counts[0] * gap_counts[1]

    # denominator
    znam = np.sum([c[0] for c in s_gaps_counts]) * np.sum([c[1] for c in s_gaps_counts])

    # gini
    g = abs(1 - (2 * first_sl + second_sl) / znam)
    return g


# load binning from file
def read_file(filename='bins.prdb'):
    """
    read binner from file
    :param filename: path to file
    :return: binner instance
    """
    with open(filename, 'rb') as inp:
        binner = pickle.load(inp)
        return binner

        # save binning to file
