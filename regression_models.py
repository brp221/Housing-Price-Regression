#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 17:00:21 2022

@author: bratislavpetkovic
"""

import pandas as pd
import numpy as np
import os
os.chdir('/Users/bratislavpetkovic/Desktop/My_Kick_Ass_Portfolio/Housing-Price-Regression/')

#_______________________________________LOAD DATA_______________________________________
test_df = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')
original_train_df = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
train_knn_df = pd.read_csv('wrangled_data/train_knn_df.csv')
train_mice_df = pd.read_csv('wrangled_data/train_mice_df.csv')
train_knn_reduced_df = pd.read_csv('wrangled_data/train_knn_reduced_df.csv')
train_mice_reduced_df = pd.read_csv('wrangled_data/train_mice_reduced_df.csv')


#_______________________________________PREP DATA_______________________________________
from sklearn.model_selection import train_test_split


X = train_knn_df.drop(columns='SalePrice')
y = train_knn_df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=33)

#_______________________________________GradientBooster_______________________________________
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(random_state=0)
gbr.fit(X_train, y_train)

gbr.predict(X_test)
gbr.score(X_test, y_test)

#_______________________________________DecisionTrees_______________________________________
from sklearn.tree import DecisionTreeRegressor

dtc = DecisionTreeRegressor(random_state=0).fit(X_train, y_train)
dtc.score(X_test,y_test)

#_______________________________________RandomForest_______________________________________
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(random_state=0).fit(X_train, y_train)
rfr.score(X_test, y_test)

#_______________________________________AdaBoost_______________________________________
from sklearn.ensemble import AdaBoostRegressor

adabr = AdaBoostRegressor(random_state=0).fit(X_train, y_train)
adabr.score(X_test, y_test)
#_______________________________________NeuralNetwork_______________________________________
from sklearn.neural_network import MLPRegressor

regr = MLPRegressor(random_state=1, max_iter=10000, solver='lbfgs', learning_rate='adaptive').fit(X_train, y_train)
regr.score(X_test, y_test)


#_______________________________________MODEL EVALUATION_______________________________________

from varname import nameof
import uuid
import seaborn as sns 

def poly_model(algo_obj, data, random_state, test_size=0.25, ):
    X = data.drop(columns='SalePrice')
    y = data['SalePrice'] 
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=random_state, test_size=test_size)
    algo_obj.fit(X_train, y_train)
    return (algo_obj.score(X_test, y_test)), algo_obj.predict(X_test)

def append_data(data_dict, algo_name, input_data, n_estimators, random_state, loss_criterion, test_size, score, y_pred_id):
    data_dict['algorithm'].append(algo_name)
    data_dict['input_data'].append(input_data)
    data_dict['n_estimators'].append(n_estimators)
    data_dict['random_state'].append(random_state)
    data_dict['loss/criterion'].append(loss_criterion)
    data_dict['test_size'].append(test_size)
    data_dict['score'].append(score)
    data_dict['y_pred_id'].append(y_pred_id)
    return data_dict

eval_model_dict = {"algorithm": [], "input_data": [], 'n_estimators' : [], 'random_state' : [] , 'loss/criterion' : [], 'test_size' : [] , 'score' : [], 'y_pred_id' : [] }
y_pred_dict = {}

data_sets = [train_knn_df, train_knn_reduced_df, train_mice_df, train_mice_reduced_df]
data_sets_name = ['train_knn_df', 'train_knn_reduced_df', 'train_mice_df', 'train_mice_reduced_df']
random_state = np.random.choice(40, 1)[0]
dataset_index=0

# PARAMETER TUNING >B-}
for dataset in data_sets:
    print('dataset : ', data_sets_name[dataset_index])
    for test_size in [0.15,0.2,0.25,0.3,0.35]:
        print('test_size : ', test_size)
        
        # DecisionTree
        for criterion in ["mse", "friedman_mse", "mae", "poisson"]:
            DecisionTreeR = DecisionTreeRegressor(random_state=random_state, criterion = criterion)
            model_acc, y_pred = poly_model(DecisionTreeR, dataset, random_state, test_size)
            y_pred_id = str(uuid.uuid1())
            y_pred_dict[y_pred_id] = y_pred
            eval_model_dict = append_data(eval_model_dict, nameof(DecisionTreeR), data_sets_name[dataset_index], 'NA', random_state, criterion, test_size,model_acc, y_pred_id )
        
        
        for n_estimator in [75,100,200,500, 1000]:
            print('n_estimator : ', n_estimator)

            # Random Forest
            for criterion in ["squared_error", "absolute_error", "friedman_mse", "poisson"]:
                RandomForestR = RandomForestRegressor(random_state=random_state, n_estimators=n_estimator)
                model_acc, y_pred = poly_model(RandomForestR, dataset, random_state, test_size)
                y_pred_id = str(uuid.uuid1())
                y_pred_dict[y_pred_id] = y_pred
                eval_model_dict = append_data(eval_model_dict, nameof(RandomForestR), data_sets_name[dataset_index], n_estimator, random_state, criterion, test_size,model_acc, y_pred_id )
                
            # AdaBoost
            for loss in ['linear', 'square', 'exponential']:
                AdaBoosterR = AdaBoostRegressor(random_state=random_state, n_estimators=n_estimator, loss=loss)
                model_acc, y_pred = poly_model(AdaBoosterR, dataset, random_state, test_size)
                y_pred_id = str(uuid.uuid1())
                y_pred_dict[y_pred_id] = y_pred
                eval_model_dict = append_data(eval_model_dict, nameof(AdaBoosterR), data_sets_name[dataset_index], n_estimator, random_state, criterion, test_size,model_acc, y_pred_id )
            
            
            # GradientBoosts
            for loss in ['ls','lad', 'huber', 'quantile']:
                params = {
                    "n_estimators": n_estimator,
                    "max_depth": 4,
                    "min_samples_split": 5,
                    "learning_rate": 0.01,
                    "loss": loss,
                    'random_state': random_state
                }
                                
                
                GradientBoostR = GradientBoostingRegressor(**params )
                model_acc, y_pred = poly_model(GradientBoostR, dataset, random_state, test_size)
                y_pred_id = str(uuid.uuid1())
                y_pred_dict[y_pred_id] = y_pred
                eval_model_dict = append_data(eval_model_dict, nameof(GradientBoostR), data_sets_name[dataset_index], n_estimator, random_state, loss, test_size,model_acc, y_pred_id )
                        
    dataset_index=dataset_index+1

models_df = pd.DataFrame.from_dict(eval_model_dict).sort_values(by='score', ascending=False)

# GET BEST MODELS BY MODEL 
idx = models_df.groupby(['algorithm'])['score'].transform(max) == models_df['score']
best_tuned_algos = models_df[idx] 


# RECREATING MOST ACCURATE MODEL
params = {
    "n_estimators": 1000,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": 'lad',
    'random_state': 39
}
GradientBoostR = GradientBoostingRegressor(**params )
X_actual = train_knn_df.drop(columns='SalePrice')
y_actual = train_knn_df['SalePrice'] 
X_train, X_test, y_train, y_test = train_test_split(X_actual,y_actual, random_state=random_state, test_size=test_size)
GradientBoostR.fit(X_train, y_train)
y_pred_winner = GradientBoostR.predict(X_actual)

# Feature Importance Interpretation 
features_df = pd.DataFrame.from_dict({'feature':list(X_train.columns), 'score':list(GradientBoostR.feature_importances_)}).sort_values(by='score', ascending=False)
features_plot = sns.barplot(data=features_df.head(13), x="feature", y="score")
features_plot.get_figure().savefig('check_me_out/feature_importances.png')


# Accuracy Interpretation
regression_plot = sns.regplot(x= y_actual, y= y_pred_winner)
regression_plot.get_figure().savefig('check_me_out/best_model_regression_plot.png')








