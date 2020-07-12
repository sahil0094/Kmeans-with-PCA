Supervised Learning

count plot
sns.set(style="darkgrid")
sns.set_palette("hls", 3)
fig, ax = plt.subplots(figsize=(20,5))
ax = sns.countplot(x="Contract", hue="Churn", data=churn_df)

for p in ax.patches:
    #print(p)
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/churn_df.shape[0]),
            ha="center")

Cramers V
chi2_contingency([ct.iloc[0].values,ct.iloc[1].values,ct.iloc[2].values])
np.sqrt(1184.5965/churn_df.shape[0])

model building
def modelis(X, method, fold):
    method = str(method)
    if method == 'minmax':
        mn = MinMaxScaler()
        X_std = mn.fit_transform(X)
        
    elif method == 'power':
        pt = PowerTransformer()
        X_std = pt.fit_transform(X)

    elif method == 'robust':
        rc = RobustScaler()
        X_std = rc.fit_transform(X)        
        
    else:
        sc = StandardScaler()
        X_std = sc.fit_transform(X)
        
    gb = GradientBoostingClassifier()
    nb = GaussianNB()
    dt = DecisionTreeClassifier(criterion = 'gini', random_state=0) # Specifying random state is to 0
    rg_dt = DecisionTreeClassifier(criterion = 'gini', random_state=0, max_depth=7)    
    knn = KNeighborsClassifier(weights='distance')
    
    rf = RandomForestClassifier(n_estimators= 10, criterion= 'entropy', random_state=0)
    dt_bag = BaggingClassifier(base_estimator = dt, n_jobs=-1, random_state=0)
    dt_rg_bag = BaggingClassifier(base_estimator = rg_dt, n_jobs=-1, random_state=0)    
    dt_boost = AdaBoostClassifier(base_estimator= dt, n_estimators= 10, random_state=0)
    #rf_boost = AdaBoostClassifier(base_estimator= rf, n_estimators= 100, random_state=0)

    models = []

    models.append(('Gradient_Boosting', gb))
    models.append(('Naive_Bayes', nb))
    models.append(('Decision_Tree', dt))
    models.append(('KNN', knn))
    #models.append(('KNN', knn))
    models.append(('Bagged_Decision_Tree',dt_bag))
    models.append(('Random_Forest', rf))
    models.append(('Boosted_Decision_Tree', dt_boost))
    models.append(('Bagged_RG_DT', dt_rg_bag))

    result= []
    names = []
    for name,model in models:
        if fold == 'kf':
            skf = KFold(shuffle=True, n_splits=5, random_state=0)
        elif fold == 'skf':
            skf = StratifiedKFold(shuffle=True, n_splits=5, random_state=0)
        else:
            skf = KFold(shuffle=True, n_splits=5, random_state=0)
        cv_results = cross_val_score(model, X_std, y, cv=skf, scoring='f1_weighted')
        result.append(cv_results)
        names.append(name)
        print('%s: %f, (%f)' %(name, 1-np.mean(cv_results), np.var(cv_results, ddof=1)))
    fig = plt.figure()
    plt.grid()
    fig.suptitle('Algo Comparison')
    ax = fig.add_subplot(111)
    plt.xticks(rotation=90)
    plt.boxplot(result)
    ax.set_xticklabels(names)
    plt.show()
