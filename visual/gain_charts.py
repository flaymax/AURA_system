def df_style(val):
    return "font-weight: bold"

def get_cum_percentile_buckets(df: pd.DataFrame, 
                               column_to_sort: str, 
                               target_column: str,
                               baseline_ar: float) -> tuple[pd.DataFrame, dict]:
    df = df.sort_values(by=column_to_sort)
    buckets = [i*5 + 5 for i in range(15) ] + [baseline_ar]
    target = []
    cum_bucket = dict()
    counts = []
    for bucket in buckets:
        target_mean = df.apply(lambda x: x.head(int(len(x)*0.01*bucket))).reset_index(drop=True)[target_column].mean()
        count = df.apply(lambda x: x.head(int(len(x)*0.01*bucket))).reset_index(drop=True)[target_column].shape[0]
        counts.append(count)
        target.append(target_mean)
        cum_bucket[bucket] = target_mean
    return pd.DataFrame(columns = ['perc_cum_buckets', 'count', target_column], 
                        data =zip(buckets, counts, target)).set_index(keys='perc_cum_buckets'), cum_bucket

def get_cumul_month_percentile_buckets(df_init: pd.DataFrame, 
                                     column_to_sort: str, 
                                     column_to_sort2: str,
                                     target_column: str, 
                                     month_column: str, 
                                     is_pure: int,
                                     baseline_ar: float) -> pd.DataFrame:
    arrays = [sorted(list(df_init[month_column].unique())*3),  
          ["кол-во", "было", "стало"]*len(df_init[month_column].unique())]
    tuples = list(zip(*arrays))
    buckets = [i*5 + 5 for i in range(15)] + [baseline_ar]
    index = pd.MultiIndex.from_tuples(tuples, names=[month_column, "ar"])

  
    df = pd.DataFrame(index = get_cum_percentile_buckets(df_init[(df_init[month_column]==df_init[month_column].unique()[0])], 
                                       column_to_sort=column_to_sort, 
                                       target_column=target_column,
                                       baseline_ar=baseline_ar)[0].rename({target_column: df_init[month_column].unique()[0]}, axis=1).index, 
                                       columns = ['tmp'], data = [])
    for month in sorted(df_init[month_column].unique()):
        if is_pure==1:
            
            df = df.join(get_cum_percentile_buckets(df_init[(df_init['is_pure']==1)  & (df_init[month_column]==month)], 
                                           column_to_sort=column_to_sort, 
                                           target_column=target_column,
                                            baseline_ar=baseline_ar)[0].rename({'count': 'count_{}'.format(month), target_column: month}, axis=1))
            try:
                df = df.join(get_cum_percentile_buckets(df_init[(df_init['is_pure']==1)  & (df_init[month_column]==month)], 
                                   column_to_sort=column_to_sort2, 
                                   target_column=target_column,
                                    baseline_ar=baseline_ar)[0].rename({target_column: month+'x'}, 
                                                                          axis=1).drop(['count'], axis=1))
            except TypeError:
                df = df.join(get_cum_percentile_buckets(df_init[(df_init['is_pure']==1)  & (df_init[month_column]==month)], 
                                   column_to_sort=column_to_sort2, 
                                   target_column=target_column,
                                    baseline_ar=baseline_ar)[0].rename({target_column: str(month)+'x'}, 
                                                                          axis=1).drop(['count'], axis=1))
        else:
            df = df.join(get_cum_percentile_buckets(df_init[(df_init[month_column]==month)], 
                               column_to_sort=column_to_sort, 
                               target_column=target_column,
                                baseline_ar=baseline_ar)[0].rename({'count': 'count_{}'.format(month), target_column: month}, axis=1))
            try:
                df = df.join(get_cum_percentile_buckets(df_init[(df_init[month_column]==month)], 
                       column_to_sort=column_to_sort2, 
                       target_column=target_column,
                       baseline_ar=baseline_ar)[0].rename({target_column: month+'x'}, axis=1).drop(['count'], axis=1))
            except TypeError:
                df = df.join(get_cum_percentile_buckets(df_init[(df_init[month_column]==month)], 
                       column_to_sort=column_to_sort2, 
                       target_column=target_column,
                       baseline_ar=baseline_ar)[0].rename({target_column: str(month)+'x'}, axis=1).drop(['count'], axis=1))

        df[month] = df[month]
        #* 100 // 0.1 * 0.1 
        try:
            df[month+'x'] = df[month+'x']
            # * 100 // 0.1 * 0.1
        except TypeError:
            df[str(month)+'x'] = df[str(month)+'x']
            # * 100 // 0.1 * 0.1
        df['count_{}'.format(month)] = df['count_{}'.format(month)].astype(int)

    df.drop(['tmp'], inplace=True, axis=1)
    
    df_res = pd.DataFrame(df.values,index=buckets, columns=index).sort_index()
    bold_row = pd.IndexSlice[df_res.index[df_res.index == baseline_ar], :]
    df_res = df_res.style.applymap(df_style, subset=bold_row) #.background_gradient(cmap='Oranges') # TODO
 
    return display(df_res)