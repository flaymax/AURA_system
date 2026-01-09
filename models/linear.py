
def check_feature_by_month(df, algo='lr', target='fpd10', params={}):
    X_train = df[df['sample_type']!=2]
    y_train = df[df['sample_type']!=2][target]

#     X_test = df[df['sample_type']==1]
#     y_test = df[df['sample_type']==1][target]
    
    X_valid = df[df['sample_type']==2]
    y_valid = df[df['sample_type']==2][target]
    
    if algo == 'lr':
        clf = LogisticRegression(random_state=42)
    else:
        clf = RandomForestClassifier(**params)
    clf.fit(X_train.iloc[:, :-4], y_train)
    preds_prob = clf.predict_proba(X_train.iloc[:, :-4])[:, 1]
    aucroc = roc_auc_score(y_train, preds_prob)
    gini = 2*roc_auc_score(y_train, preds_prob) - 1
    print("train gini : ", gini)
#     preds_prob = clf.predict_proba(X_test.iloc[:, :-4])[:, 1]
#     aucroc = roc_auc_score(y_test, preds_prob)
#     gini = 2*roc_auc_score(y_test, preds_prob) - 1
#     print("test gini : ", gini)
    preds_prob = clf.predict_proba(X_valid.iloc[:, :-4])[:, 1]
    aucroc = roc_auc_score(y_valid, preds_prob)
    gini = 2*roc_auc_score(y_valid, preds_prob) - 1
    print("valid gini : ", gini)
    
    # gini train by month

    print("\n","train gini by month")
    for month in X_train["month_year"].sort_values().unique():
        preds_prob = clf.predict_proba(X_train[X_train["month_year"] == month].iloc[:, :-4])[:, 1]
        y_true =  X_train[X_train["month_year"] == month][target]
        gini = 2*roc_auc_score(y_true, preds_prob) - 1
        print(month, round(gini,3), y_true.shape[0])
        
#     # gini test by month
#     print("\n", "test gini by month")
#     for month in X_test["month_year"].sort_values().unique():
#         preds_prob = clf.predict_proba(X_test[X_test["month_year"] == month].iloc[:, :-4])[:, 1]
#         y_true =  X_test[X_test["month_year"] == month][target]
#         gini = 2*roc_auc_score(y_true, preds_prob) - 1
#         print(month, round(gini,3), y_true.shape[0])
    
    # gini valid by month
    print("\n", "valid gini by month")
    for month in X_valid["month_year"].sort_values().unique():
        preds_prob = clf.predict_proba(X_valid[X_valid["month_year"] == month].iloc[:, :-4])[:, 1]
        y_true =  X_valid[X_valid["month_year"] == month][target]
        gini = 2*roc_auc_score(y_true, preds_prob) - 1
        print(month, round(gini,3), y_true.shape[0])
#     
    # gini train by month is_pure
#     print("\n", "train gini by month with is_pure")
    for month in X_train["month_year"].sort_values().unique():
        preds_prob = clf.predict_proba(X_train[(X_train["month_year"] == month) 
                                               & (X_train["is_pure"] == 1)].iloc[:, :-4])[:, 1]
        y_true =  X_train[(X_train["month_year"] == month) & (X_train["is_pure"] == 1)][target]
        gini = 2*roc_auc_score(y_true, preds_prob) - 1
#         print(month, round(gini,3), y_true.shape[0])
        
    # gini test by month is_pure
#     print("\n", "test gini by month with is_pure")
#     for month in X_test["month_year"].sort_values().unique():
#         preds_prob = clf.predict_proba(X_test[(X_test["month_year"] == month) 
#                                               & (X_test["is_pure"] == 1)].iloc[:, :-4])[:, 1]
#         y_true =  X_test[(X_test["month_year"] == month) & (X_test["is_pure"] == 1)][target]
#         gini = 2*roc_auc_score(y_true, preds_prob) - 1
#         print(month, round(gini,3), y_true.shape[0])
    
    # gini valid by month is_pure
#     print("\n", "valid gini by month with is_pure")
    for month in X_valid["month_year"].sort_values().unique():
        preds_prob = clf.predict_proba(X_valid[(X_valid["month_year"] == month) 
                                               & (X_valid["is_pure"] == 1)].iloc[:, :-4])[:, 1]
        y_true =  X_valid[(X_valid["month_year"] == month) & (X_valid["is_pure"] == 1)][target]
        gini = 2*roc_auc_score(y_true, preds_prob) - 1
#         print(month, round(gini,3), y_true.shape[0])
    
    # coeff
    if algo == 'lr':
        print("\n", "LogReg coeff. : ")
        for i,j in zip(X_train.iloc[:, :-4].columns, clf.coef_[0]):
            print(i,j)
                     
    return None

def check_graphviz_plot(df, target='fpd10', max_depth=3):
    X_train = df[df['sample_type']!=2]
    y_train = df[df['sample_type']!=2][target]

#     X_test = df[df['sample_type']==1]
#     y_test = df[df['sample_type']==1][target]
    
    X_valid = df[df['sample_type']==2]
    y_valid = df[df['sample_type']==2][target]
    
    dct = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=300)
    dct.fit(X_train.iloc[:, :-4], y_train)
    preds_prob = dct.predict_proba(X_train.iloc[:, :-4])[:, 1]
    aucroc = roc_auc_score(y_train, preds_prob)
    gini = 2*roc_auc_score(y_train, preds_prob) - 1
    print("train gini : ", gini)
#     preds_prob = dct.predict_proba(X_test.iloc[:, :-4])[:, 1]
#     aucroc = roc_auc_score(y_test, preds_prob)
#     gini = 2*roc_auc_score(y_test, preds_prob) - 1
#     print("test gini : ", gini)
    preds_prob = dct.predict_proba(X_valid.iloc[:, :-4])[:, 1]
    aucroc = roc_auc_score(y_valid, preds_prob)
    gini = 2*roc_auc_score(y_valid, preds_prob) - 1
    print("valid gini : ", gini)
    # gini train by month
#     print("\n","train gini by month")
    for month in X_train["month_year"].sort_values().unique():
        preds_prob = dct.predict_proba(X_train[X_train["month_year"] == month].iloc[:, :-4])[:, 1]
        y_true =  X_train[X_train["month_year"] == month][target]
        gini = 2*roc_auc_score(y_true, preds_prob) - 1
#         print(month, round(gini,3), y_true.shape[0])
        
#     # gini test by month
#     print("\n", "test gini by month")
#     for month in X_test["month_year"].sort_values().unique():
#         preds_prob = dct.predict_proba(X_test[X_test["month_year"] == month].iloc[:, :-4])[:, 1]
#         y_true =  X_test[X_test["month_year"] == month][target]
#         gini = 2*roc_auc_score(y_true, preds_prob) - 1
#         print(month, round(gini,3), y_true.shape[0])
    
    # gini valid by month
#     print("\n", "valid gini by month")
    for month in X_valid["month_year"].sort_values().unique():
        preds_prob = dct.predict_proba(X_valid[X_valid["month_year"] == month].iloc[:, :-4])[:, 1]
        y_true =  X_valid[X_valid["month_year"] == month][target]
        gini = 2*roc_auc_score(y_true, preds_prob) - 1
#         print(month, round(gini,3), y_true.shape[0])
    
    
    # gini train by month is_pure
    print("\n", "train gini by month with is_pure")
    for month in X_train["month_year"].sort_values().unique():
        preds_prob = dct.predict_proba(X_train[(X_train["month_year"] == month) 
                                               & (X_train["is_pure"] == 1)].iloc[:, :-4])[:, 1]
        y_true =  X_train[(X_train["month_year"] == month) & (X_train["is_pure"] == 1)][target]
        gini = 2*roc_auc_score(y_true, preds_prob) - 1
        print(month, round(gini,3), y_true.shape[0])
        
#     # gini test by month is_pure
#     print("\n", "test gini by month with is_pure")
#     for month in X_test["month_year"].sort_values().unique():
#         preds_prob = dct.predict_proba(X_test[(X_test["month_year"] == month) 
#                                               & (X_test["is_pure"] == 1)].iloc[:, :-4])[:, 1]
#         y_true =  X_test[(X_test["month_year"] == month) & (X_test["is_pure"] == 1)][target]
#         gini = 2*roc_auc_score(y_true, preds_prob) - 1
#         print(month, round(gini,3), y_true.shape[0])
    
    # gini valid by month is_pure
    print("\n", "valid gini by month with is_pure")
    for month in X_valid["month_year"].sort_values().unique():
        preds_prob = dct.predict_proba(X_valid[(X_valid["month_year"] == month) 
                                               & (X_valid["is_pure"] == 1)].iloc[:, :-4])[:, 1]
        y_true =  X_valid[(X_valid["month_year"] == month) & (X_valid["is_pure"] == 1)][target]
        gini = 2*roc_auc_score(y_true, preds_prob) - 1
        print(month, round(gini,3), y_true.shape[0])
        
    dot_data = export_graphviz(dct, feature_names=X_train.iloc[:, :-4].columns, filled=True, proportion=True, rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    display(Image(graph.create_png()))
    graph = graphviz.Source(dot_data)
    graph
def rounded_mean(x):
    return round(np.mean(x),2)