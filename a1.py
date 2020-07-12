Supervised Learning

library
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

correlation plot
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
#cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask,cmap='RdYlGn', vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)
plt.show()

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
ct=pd.crosstab(index=churn_df.Contract,columns=churn_df.Churn)
ct
chi2_contingency([ct.iloc[0].values,ct.iloc[1].values,ct.iloc[2].values])
np.sqrt(1184.5965/churn_df.shape[0])

Feature Selection
Backward Elimination: In backward elimination, we start with all the features and removes the least significant feature at each iteration which improves the performance of the model. We repeat this until no improvement is observed on removal of features.

Recursive Feature elimination: It is a greedy optimization algorithm which aims to find the best performing feature subset. It repeatedly creates models and keeps aside the best or the worst performing feature at each iteration. It constructs the next model with the left features until all the features are exhausted. It then ranks the features based on the order of their elimination.

Backward
#Backward Elimination
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)

RFE
#no of features
nof_list=np.arange(1,13)            
high_score=0
#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))

cols = list(X.columns)
model = LinearRegression()
#Initializing RFE model
rfe = RFE(model, 10)             
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  
#Fitting the data to model
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)

Lasso
Regularization methods are the most commonly used embedded methods which penalize a feature given a coefficient threshold.
Lasso regression performs L1 regularization which adds penalty equivalent to absolute value of the magnitude of coefficients.

reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

imp_coef = coef.sort_values(ascending=False)
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")

VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor

[variance_inflation_factor(X.values, j) for j in range(1, X.shape[1])]

def calculate_vif(x):
    thresh = 5.0
    output = pd.DataFrame()
    k = x.shape[1]
    vif = [variance_inflation_factor(x.values, j) for j in range(x.shape[1])]
    for i in range(1,k):
        print("Iteration no.")
        print(i)
        print(vif)
        a = np.argmax(vif)
        print("Max VIF is for variable no.:")
        print(a)
        if vif[a] <= thresh :
            break
        if i == 1 :          
            output = x.drop(x.columns[a], axis = 1)
            vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]
        elif i > 1 :
            output = output.drop(output.columns[a],axis = 1)
            vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]
    return(output)

model building
def modelcomp(X_train,Y_train,fold):
    
    GBoost=GradientBoostingRegressor(n_estimators=22,random_state=0)
    RF=RandomForestRegressor(n_estimators=1,random_state=0,criterion='mse')
    DT=DecisionTreeRegressor(max_depth=5,criterion='mse',min_samples_leaf=7,random_state=0)
    Bag_DT=BaggingRegressor(base_estimator=DT,n_estimators=1,random_state=0)
    AB_RF=AdaBoostRegressor(base_estimator=RF,n_estimators=9,random_state=0)
    KNN=KNeighborsRegressor(n_neighbors=4,weights='distance')
    AB=AdaBoostRegressor(n_estimators=20,random_state=0)
    AB_DT=AdaBoostRegressor(base_estimator=DT,n_estimators=4,random_state=0)
    
    

    models = []

    models.append(('Gradient_Boosting', GBoost))
    models.append(('Random_Forest', RF))
    models.append(('Decision_Tree', DT))
    models.append(('Bagged_Decision_Tree',Bag_DT))
    models.append(('AdaBoosted_RandomForest', AB_RF))
    models.append(('KNN', KNN))
    models.append(('AdaBoost',AB))
    models.append(('AdaBoost_DecisionTree', AB_DT))

    result= []
    names = []
    for name,model in models:
        
        kf = KFold(shuffle=True, n_splits=5, random_state=0)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kf, scoring='neg_mean_squared_error')
        result.append(cv_results)
        names.append(name)
        print('%s: %f, (%f)' %(name, np.mean(np.sqrt(np.abs(cv_results))), np.var(np.sqrt(np.abs(cv_results)), ddof=1)))
        print(f'Mean of bias & variance error is {((np.mean(np.sqrt(np.abs(cv_results))))+np.var(np.sqrt(np.abs(cv_results)), ddof=1))/2}')
        print()
    fig = plt.figure()
    plt.grid()
    plt.style.use('dark_background')
    fig.suptitle('Algo Comparison')
    ax = fig.add_subplot(111)
    plt.xticks(rotation=90)
    plt.boxplot(result)
    ax.set_xticklabels(names)
    plt.show()
	
XGBoost
cv_score=[]
predictions=np.zeros(X_test.shape[0])
K_fold=KFold(n_splits=5,shuffle=True,random_state=1)
i=1
for train_index,val_index in K_fold.split(X_train,Y_train):
    #print(train_index)
    #print(type(X_train))
    x_train,x_val=X_train.loc[train_index],X_train.loc[val_index]
    y_train,y_val=Y_train[train_index],Y_train[val_index]
    
    cls_xgb=xgboost.XGBRegressor(base_score=.5, booster='gbtree', colsample_bylevel=1, colsample_bytree=0.8,
                                  gamma=2, learning_rate=0.05, max_delta_step=0, max_depth=7, min_child_weight=7,
                                  missing=None, n_estimators=150, n_jobs=-1, objective='reg:linear',
                                  random_state=1, reg_lambda=1, scale_pos_weight=1, seed=None,
                                  subsample=0.85,verbosity=2,silent=True)
    
    cls_xgb.fit(x_train,y_train)
    score_train = np.sqrt(mse(y_train,cls_xgb.predict(x_train)))
    score_val = np.sqrt(mse(y_val,cls_xgb.predict(x_val)))
    print('{} of KFold {}'.format(i,K_fold.n_splits))
    print('Training rmse score:',score_train)
    print('Validation rmse score:',score_val)
    print("-----------------------------------")
    cv_score.append(score_val)    
    predictions += cls_xgb.predict(X_test)/ K_fold.n_splits
    i+=1
print ("Mean Cross Validation rmse Score:{}".format(np.array(cv_score).mean()))
print ("Variance of Cross Validation rmse Score:{}".format(np.array(cv_score).var()))
