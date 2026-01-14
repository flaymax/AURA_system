class Binning:
    def __init__(self, x, y, variable_name, power=0, settings=None, binner_type=BinnerType.IV, good_mark=0, bad_mark=1):
        self.fit(x, y, variable_name, power, settings, binner_type, good_mark, bad_mark)


    def fit(self, x, y, variable_name, power=0, settings=None, binner_type=BinnerType.IV, good_mark=0, bad_mark=1):
        if good_mark == bad_mark:
            raise Exception('Classes cant be equal!!!')
        self._good_v = good_mark
        self._bad_v = bad_mark
        self._binner_type = binner_type
        check_mono = True
        min_leaf_ratio = 0.1
        if settings is not None:
            check_mono = settings._monotone
            min_leaf_ratio = settings._min_leaf_ratio

        # initialize results
        best_gaps = []
        best_gaps_shares = []
        best_gaps_woe = []
        best_iv = 0
        best_gaps_counts = []
        best_r2 = 0
        best_gaps_counts_shares = []
        best_gaps_avg = []
        best_hhi = 1

        # create grouping variable: bad / good
        y_gr = y.copy()
        y_gr[y > np.average(y)] = 1
        y_gr[y <= np.average(y)] = 0

        # rename series for easier access later
        x.rename("x", inplace=True)
        y.rename("y", inplace=True)
# #         if not isinstance(y, pd.Series):
# #             y = pd.Series(y, name="y")
# #         else:
# #             y.rename("y", inplace=True)
#         if isinstance(y, pd.DataFrame):
#             # If y is a DataFrame, consider converting it to a Series or selecting a single column
#             if y.shape[1] == 1:
#                 y = y.iloc[:, 0]  # Select the first column as a Series
#             else:
#                 raise ValueError("y should be a Series or single-column DataFrame")

#         # Check if y is empty or a non-Series object before converting
#         if not isinstance(y, pd.Series):
#             if y.empty:
#                 raise ValueError("y is empty; cannot proceed with empty target values")
#             else:
#                 y = pd.Series(y, name="y")
#         else:
#             y.rename("y", inplace=True)


        y_gr.rename("y_gr", inplace=True)
        all_set = pd.concat([x, y, y_gr], axis=1)

        data_goods = all_set[y_gr == self._good_v]
        data_bads = all_set[y_gr == self._bad_v]

        # non-NaN data
        clear_data = all_set.dropna()

        # handle NaN values
        if x.isnull().sum() > 0:
            dirty_set = all_set[pd.isnull(all_set['x'])]
            dirty_bads = dirty_set[dirty_set['y_gr'] == self._bad_v]
            dirty_goods = dirty_set[dirty_set['y_gr'] == self._good_v]
            dirty_woe, dirty_goods_share, dirty_bads_share = _calc_WoE(
                gap_bads=dirty_bads,
                bads=data_bads,
                gap_goods=dirty_goods,
                goods=data_goods
            )
            dirty_iv = (dirty_goods_share - dirty_bads_share) * dirty_woe
            dirty_counts_shares = (len(dirty_goods)) / (len(dirty_bads) + 0.000000001)

        # build trees with different depths, searching for the best split on clean (non-NaN) data
        for depth in range(1, power + 1):

            # build decision tree
            dt = tree.DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=max(int(min_leaf_ratio * len(all_set['y'])), 1)
            )
            dt.fit(clear_data['x'].values.reshape(-1, 1), clear_data['y_gr'])

            # save resulting splits
            gaps = self._get_gaps(dt)

            # calculate split metrics
            gaps_shares, gaps_woe, gaps_counts, gaps_counts_shares, gaps_avg = self._gaps_metrics(
                gaps,
                clear_data,
                all_set[all_set['y_gr'] == self._good_v],
                all_set[all_set['y_gr'] == self._bad_v]
            )

            temp_mono = False

            if self._binner_type == BinnerType.IV:
                temp_mono = _is_monotonik(vals=gaps_counts_shares)

            if self._binner_type == BinnerType.R2:
                temp_mono = _is_monotonik(gaps_avg)

            is_mono = temp_mono or not check_mono  # monotonicity condition

            # add missing values bin
            if x.isnull().sum() > 0:
                gaps.append([None, None])
                gaps_shares.append([dirty_goods_share, dirty_bads_share])
                gaps_counts.append([len(dirty_goods), len(dirty_bads)])
                gaps_woe.append(dirty_woe)
                gaps_counts_shares.append(dirty_counts_shares)

            # GINI of the split
            gini = _calc_gini(gaps_counts)

            # HHI
            hhi = _calc_HHI(gaps_counts)

            # IV over all bins
            ivs = [(gs[0] - gs[1]) * gw for gs, gw in zip(gaps_shares, gaps_woe)]
            iv = np.sum(ivs)

            # case: maximize R2
            if self._binner_type == BinnerType.R2:
                # create candidate binning object
                b = Binning(gaps, gaps_woe, iv, gaps_shares, gaps_counts, gini, gaps_counts_shares, gaps_avg)
                # overall R2
                r2 = _calc_r2(x, y, b)
                # select best by R2
                if r2 > best_r2 and is_mono:
                    best_gaps = gaps
                    best_gaps_shares = gaps_shares
                    best_gaps_woe = gaps_woe
                    best_gaps_counts = gaps_counts
                    best_iv = iv
                    best_r2 = r2
                    best_gaps_counts_shares = []
                    best_gaps_avg = gaps_avg
                    best_hhi = hhi

            # case: maximize IV
            if self._binner_type == BinnerType.IV:
                # select best by IV
                if iv > best_iv and is_mono:
                    best_gaps = gaps
                    best_gaps_shares = gaps_shares
                    best_gaps_woe = gaps_woe
                    best_gaps_counts = gaps_counts
                    best_gaps_counts_shares = gaps_counts_shares
                    best_iv = iv
                    best_hhi = hhi

        # best split
        self._gaps = best_gaps
        self._woes = best_gaps_woe
        self._iv = best_iv
        self._shares = best_gaps_shares
        self._counts = best_gaps_counts
        self._gini = gini
        self._gaps_counts_shares = best_gaps_counts_shares
        self._gaps_avg = best_gaps_avg
        self._hhi = best_hhi
        self._r2 = best_r2
        self._name = variable_name

    # Apply bins to data – returns WOE factors
    def transform(self, data_in):
        """
        Transform data
        :param data_in: Data to be transformed
        :param exclude: List of excluded variables (will be ignored during transformation)
        :return: Transformed data
        """
        data_out = data_in.copy()
        data_out.name = self._name + '_woe'

        for gap, value in zip(self._gaps, self._woes):
            if gap[0] is None:
                data_out.loc[(data_in.isnull())] = value
            else:
                data_out.loc[(data_in >= gap[0]) & (data_in < gap[1])] = value
        return data_out

    # Extract intervals from the tree
    def _get_gaps(self, estimator):
        maxval = 100000000000000
        # extract split thresholds without duplicates
        threshold = _delete_duplicates(estimator.tree_.threshold)
        # remove artificial boundary -2
        if -2 in threshold:
            threshold.remove(-2)
        # add upper and lower bounds to close extreme intervals
        threshold.append(maxval)
        threshold.append(maxval * -1)
        # select all boundaries and sort them
        threshold = list(set(threshold))
        threshold.sort()
        # build intervals from boundaries
        gaps = []
        for i in range(len(threshold) - 1):
            gaps.append([threshold[i], threshold[i + 1]])
        return gaps

    # Calculate split metrics
    def _gaps_metrics(self, gaps, data, data_good, data_bad):
        woe = 0
        gaps_shares = []
        gaps_woe = []
        gaps_counts = []
        gaps_avg = []
        # for each interval
        for gap in gaps:
            # lower and upper bounds of the interval
            st = gap[0]
            end = gap[1]
            # observations in the segment
            gap_data = data[(data['x'] >= st) & (data['x'] <= end)]
            # goods
            gap_goods = gap_data[gap_data['y_gr'] == self._good_v]
            # bads
            gap_bads = gap_data[gap_data['y_gr'] == self._bad_v]

            # mean value
            gap_avg = np.average(gap_data)
            gaps_avg.append(gap_avg)

            # counts
            gaps_counts.append([len(gap_goods), len(gap_bads)])
            # WOE for the group
            woe, gap_goods_share, gap_bads_share = _calc_WoE(
                gap_goods=gap_goods,
                goods=data_good,
                gap_bads=gap_bads,
                bads=data_bad
            )
            # WOE for the split
            gaps_shares.append([gap_goods_share, gap_bads_share])
            gaps_woe.append(woe)

        gaps_counts_shares = [
            (gaps_count[0] + 0.000001) / (gaps_count[1] + 0.000001)
            for gaps_count in gaps_counts
        ]

        return gaps_shares, gaps_woe, gaps_counts, gaps_counts_shares, gaps_avg


class BinningSettings:
    _monotone = True
    _variable_name = ''
    _min_leaf_ratio = 0.1

    def __init__(self, variable_name='', monotone=True, min_leaf_ratio=0.1):
        self._monotone = monotone
        self._variable_name = variable_name
        self._min_leaf_ratio = min_leaf_ratio
