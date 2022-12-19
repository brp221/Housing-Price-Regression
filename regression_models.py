import pandas as pd
import numpy as np
import os
os.chdir('/Users/bratislavpetkovic/Desktop/My_Kick_Ass_Portfolio/Housing-Price-Regression/')

#_______________________________________LOAD DATA_______________________________________
test_df = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')
test_prepped_df = pd.read_csv('wrangled_data/test_prepped_df.csv')

original_train_df = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
train_knn_df = pd.read_csv('wrangled_data/train_knn_df.csv')
train_mice_df = pd.read_csv('wrangled_data/train_mice_df.csv')
train_knn_reduced_df = pd.read_csv('wrangled_data/train_knn_reduced_df.csv')
train_mice_reduced_df = pd.read_csv('wrangled_data/train_mice_reduced_df.csv')

random_state = np.random.choice(40, 1)[0]

#_______________________________________PREP DATA_______________________________________
from sklearn.model_selection import train_test_split


X = train_knn_df.drop(columns='SalePrice')
y = train_knn_df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=33)

#_______________________________________GradientBooster_______________________________________
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(random_state=random_state)
gbr.fit(X_train, y_train)
gbr.predict(X_test)
print(gbr.score(X_test, y_test))

#_______________________________________DecisionTrees_______________________________________
from sklearn.tree import DecisionTreeRegressor

dtc = DecisionTreeRegressor(random_state=random_state).fit(X_train, y_train)
print(dtc.score(X_test,y_test))

#_______________________________________RandomForest_______________________________________
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(random_state=random_state).fit(X_train, y_train)
print(rfr.score(X_test, y_test))

#_______________________________________AdaBoost_______________________________________
from sklearn.ensemble import AdaBoostRegressor

adabr = AdaBoostRegressor(random_state=random_state).fit(X_train, y_train)
print(adabr.score(X_test, y_test))

#_______________________________________MODEL HYPERTUNING _______________________________________

from sklearn.model_selection import GridSearchCV
X_actual = train_knn_df.drop(columns='SalePrice')
y_actual = train_knn_df['SalePrice'] 
X_train, X_test, y_train, y_test = train_test_split(X_actual, y_actual, random_state=random_state, test_size=0.35)

tune_params = {'learning_rate':[0.01, 0.1,  1.0], 
               'n_estimators' : [500,1000,2000], 
               'loss' : ['ls', 'lad', 'huber', 'quantile'],
               'min_samples_split' : [2,4], 
               'min_samples_leaf' : [1,2]
               }
GradientBoostR = GradientBoostingRegressor(random_state = random_state)

model_selection_ = GridSearchCV(GradientBoostR, tune_params, verbose=1, n_jobs=2)
model_selection_.fit(X_train, y_train)

models_df = pd.DataFrame.from_dict(model_selection_.cv_results_).sort_values(by='mean_test_score', ascending=False)
models_df.to_csv('check_me_out/GradBoostR_Models.csv', index=False)


#_______________________________________FINAL MODEL_______________________________________
optim_params = {'learning_rate': 0.05, 
                'n_estimators' : 1000, 
                'loss' : 'huber',
                'min_samples_split' : 2, 
                'min_samples_leaf' : 2, 
                'random_state':random_state}
GradientBoostR = GradientBoostingRegressor(**optim_params)
GradientBoostR.fit(X_train, y_train)

#_______________________________________MODEL INTERPRETAION_______________________________________
# overfitting vs underfitting 
from sklearn.model_selection import cross_validate

cv_results = cross_validate(GradientBoostR, X_actual, y_actual, cv=3, return_train_score=True, return_estimator=True, scoring=('r2'))
print("TEST score   : {:.1%}".format(np.mean(cv_results['test_score'])) , " (+/-) {:.1%}".format(np.std(cv_results['test_score'])))
print("TRAIN score  : {:.1%}".format(np.mean(cv_results['train_score'])), " (+/-) {:.1%}".format(np.std(cv_results['train_score'])))


import seaborn as sns 
sns.set_theme(style="whitegrid", palette="Spectral", rc={'figure.figsize':(20,20)}, font_scale=1.2)

# Feature Importance Interpretation 
features_df = pd.DataFrame.from_dict({'feature':list(X_train.columns), 'score':list(GradientBoostR.feature_importances_)}).sort_values(by='score', ascending=False)
print(features_df.head(14))
features_plot = sns.barplot(data=features_df.head(13), x="feature", y="score")
features_plot.get_figure().savefig('check_me_out/feature_importances.png')


# Accuracy Visualization
y_pred_winner = GradientBoostR.predict(X_actual)
regression_plot = sns.regplot(x= y_actual, y= y_pred_winner)
regression_plot.set(xlabel='Actual Sale Price', ylabel='Predicted Sale Price')
regression_plot.get_figure().savefig('check_me_out/best_model_regression_plot.png')


#_______________________________________REGULARIZATION_______________________________________
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
font = {'weight' : 'bold','size'   : 18}
plt.rc('font', **font)
generic_params = {'n_estimators' : 1000, 
                'loss' : 'huber',
                'min_samples_split' : 2, 
                'min_samples_leaf' : 2, 
                'random_state':random_state}

settings = [ ("learning_rate=1 subsample=1", "orange", {"learning_rate": 1.0, "subsample": 1.0}),
            ("learning_rate=.5 subsample=1 ", "green", {"learning_rate": 0.5, "subsample": 1.0}),
            ("learning_rate=1 subsample=0.5", "blue", {"learning_rate": 1, "subsample": 0.5}),
            ("learning_rate=0.5, subsample=0.5","gray", {"learning_rate": 0.5, "subsample": 0.5}),
            ("learning_rate=0.5, max_features=2", "magenta", {"learning_rate": 0.5, "max_features": 2})]

fig = plt.figure(figsize=(16, 16))
for label, color, setting in settings:
    params = generic_params
    params.update(setting)

    clf = GradientBoostingRegressor(**params)
    clf.fit(X_train, y_train)

    # compute test set deviance
    test_deviance = np.zeros((generic_params["n_estimators"],), dtype=np.float64)

    for i, y_proba in enumerate(clf.staged_predict(X_test)):
        test_deviance[i] = mean_squared_error(y_test, y_proba)

    plt.plot( (np.arange(test_deviance.shape[0]) + 1),test_deviance,color=color,label=label)

plt.legend(loc="upper right")
plt.xlabel("Boosting Iterations")
plt.ylabel("Test Set Deviance")
plt.title('REGULARIZATION')
plt.show()
fig.savefig('check_me_out/reguralization_metric.png')


#_______________________________________FINAL PREDICTION _______________________________________
optim_params_reg = {'learning_rate': 0.05, 
                'n_estimators' : 300, 
                'loss' : 'huber',
                'min_samples_split' : 2, 
                'min_samples_leaf' : 2, 
                'random_state':random_state}
GradientBoostR = GradientBoostingRegressor(**optim_params_reg)
GradientBoostR.fit(X_train, y_train)

cv_results = cross_validate(GradientBoostR, X_actual, y_actual, cv=3, return_train_score=True, return_estimator=True, scoring=('r2'))
print("TEST score   : {:.1%}".format(np.mean(cv_results['test_score'])) , " (+/-) {:.1%}".format(np.std(cv_results['test_score'])))
print("TRAIN score  : {:.1%}".format(np.mean(cv_results['train_score'])), " (+/-) {:.1%}".format(np.std(cv_results['train_score'])))



y_test_final = GradientBoostR.predict(test_prepped_df)
submission_dict = {'Id': test_df['Id'].copy(), 'SalePrice': y_test_final}
submission_df = pd.DataFrame.from_dict(submission_dict)
submission_df.to_csv('final_result/my_submission.csv')





