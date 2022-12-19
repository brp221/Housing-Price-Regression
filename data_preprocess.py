import pandas as pd 
import numpy as np
import seaborn as sns 

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from fancyimpute import KNN, IterativeImputer 

import os
os.chdir('/Users/bratislavpetkovic/Desktop/My_Kick_Ass_Portfolio/Housing-Price-Regression/')

test_df = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')
original_train_df = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')

print(original_train_df.info())
print(test_df.info())

continuous_columns = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 
                'BsmtUnfSF' , '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 
                'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
                'MiscVal', 'MoSold', 'YrSold', 'MSSubClass', 'GarageArea', 'SalePrice', 'KitchenAbvGr', 'BedroomAbvGr', 'TotalBsmtSF' ]


#_______________________________________WRANGLING + IMPUTATION + DIMENSIONALITY REDUCTION_______________________________________
def wrangle_data(data, imputation_method='KNN', dim_reduction = True, is_test=False):
    cols_2_remove = [col for col in data.columns if data[col].isna().sum()/data.shape[0]   > 0.6 ] # rid columsn where more than 60% data is missing
    cols_2_impute = [col for col in data.columns if 0 < data[col].isna().sum()/data.shape[0]   < 0.6   ] # impute other columsn with missing data 
    columns_2_keep = list(set(data.columns) - set(cols_2_remove))
    data = data[columns_2_keep]   
    
    #  using trees so refrain from one-hot encoding
    one_hot_cols = [  'Street'  ]
    nominal_cols = [  'MSZoning', 'Neighborhood', 'Condition1', 'Condition2', 'HouseStyle', 'RoofStyle', 'LotConfig', 'BldgType', 'RoofMatl', 'Exterior1st', 
                    'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'Electrical', 'GarageType', 'SaleType', 'SaleCondition'  ]
    ordinal_cols = [  'LotShape', 'LandContour', 'Utilities', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                    'BsmtFinType2', 'HeatingQC', 'CentralAir', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive']
    remaing_cols = continuous_columns + ['Id']
    if(is_test):
        remaing_cols.remove('SalePrice')
    print('did we forget any columns?  --> ', len(set(list(one_hot_cols + nominal_cols + ordinal_cols + remaing_cols) )) != len(data.columns) )
    print('columns forgotten', set(data.columns) - set(list(one_hot_cols + nominal_cols + ordinal_cols + remaing_cols) ))
    print('columns repeated', set(list(one_hot_cols + nominal_cols + ordinal_cols + remaing_cols) ) - set(data.columns)   )
    
    #   1 Hot Encoder
    one_hot_df_copy = data[one_hot_cols+['Id']]
    prefixes = ['' for x in one_hot_cols]
    one_hot_df = pd.get_dummies(one_hot_df_copy, prefix= prefixes, columns = one_hot_cols, drop_first=True)
    print("# of New Columns Added: ", one_hot_df.shape[1] - one_hot_df_copy.shape[1])
    print("Columns Added: ", set(one_hot_df.columns) - set(one_hot_df.columns))
    print("Columns Lost: ", set(one_hot_df.columns) - set(one_hot_df.columns))
    
    #   LabelEncoding
    label_df = data[nominal_cols]
    label_df = label_df.apply(lambda series: pd.Series(LabelEncoder().fit_transform(series[series.notnull()]),index=series[series.notnull()].index))
    label_df["Id"] = data['Id'].copy()
    # print(label_df.info())
    print(label_df.Id.max())
    print(label_df.Id.min())
    print(data.Id.max())
    print(data.Id.min())
    #   OrdinalEncoding
    ordinal_df = data[ordinal_cols+remaing_cols]
    lot_shape_dic = {'IR3':0, 'IR2':1, 'IR1':2, 'Reg':3}
    land_cont_dic = {'Low':0, 'HLS':1, 'Bnk':2, 'Lvl':3}
    utilities_dic = {'ELO':0, 'NoSeWa':1, 'NoSewr':2, 'AllPub':3}
    land_slope_dic = {'Sev':0, 'Mod':1, 'Gtl':2}
    quality_cond_dic = {np.nan:-1, 'NA':-1, 'Po':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4}
    bsmnt_exp_dict = {np.nan:-1, 'No':0, 'Mn':1, 'Av':2, 'Gd':3}
    bsmnt_fin = {np.nan:-1, 'Unf':0, 'LwQ':1, 'BLQ':2, 'Rec':3, 'ALQ':4, 'GLQ':5}
    y_n_dic = {'N':0, 'Y':1}
    functional_dic = {'Sal':0, 'Sev':1, 'Maj2':2, 'Maj1':3, 'Mod':4, 'Min2':5, 'Min1':6, 'Typ':7 }
    gar_fin_dict = {np.nan:0,'NA':0, 'Unf':1, 'RFn':2, 'Fin':3}
    pav_dr_dict = {'N':0, 'P':1, 'Y':2}
    
    ordinal_df['LotShape'].replace(lot_shape_dic, inplace=True)
    ordinal_df['LandContour'].replace(land_cont_dic, inplace=True)
    ordinal_df['Utilities'].replace(utilities_dic, inplace=True)
    ordinal_df['LandSlope'].replace(land_slope_dic, inplace=True)
    ordinal_df['ExterQual'].replace(quality_cond_dic, inplace=True)
    ordinal_df['ExterCond'].replace(quality_cond_dic, inplace=True)
    ordinal_df['BsmtQual'].replace(quality_cond_dic, inplace=True) #
    ordinal_df['BsmtCond'].replace(quality_cond_dic, inplace=True) #
    ordinal_df['BsmtExposure'].replace(bsmnt_exp_dict, inplace=True)
    ordinal_df['BsmtFinType1'].replace(bsmnt_fin, inplace=True)
    ordinal_df['BsmtFinType2'].replace(bsmnt_fin, inplace=True)
    ordinal_df['HeatingQC'].replace(quality_cond_dic, inplace=True)
    ordinal_df['CentralAir'].replace(y_n_dic, inplace=True)
    ordinal_df['KitchenQual'].replace(quality_cond_dic, inplace=True)
    ordinal_df['Functional'].replace(functional_dic, inplace=True)
    ordinal_df['FireplaceQu'].replace(quality_cond_dic, inplace=True) #
    ordinal_df['GarageFinish'].replace(gar_fin_dict, inplace=True)
    ordinal_df['GarageQual'].replace(quality_cond_dic, inplace=True) #
    ordinal_df['GarageCond'].replace(quality_cond_dic, inplace=True) #
    ordinal_df['PavedDrive'].replace(pav_dr_dict, inplace=True)
    
    # Merge data back together 
    one_hot_df_cols = set(one_hot_df.columns)
    ordinal_df_cols = set(ordinal_df.columns)
    label_df_cols = set(label_df.columns)
    
    intersect = one_hot_df_cols.intersection(ordinal_df_cols,label_df_cols )
    print("One Hot Encoded: ", one_hot_df_cols)
    print("Ordinal Encoded: ", ordinal_df_cols)
    print("Label Encoded: ", label_df_cols)
    
    data = one_hot_df.merge(ordinal_df, how="outer", on="Id")
    data = data.merge(label_df, how="outer", on="Id")
    print(data.info())
    data = data.drop(columns=['Id'])
    print(data.shape)
    #_______________________________________IMPUTATION_______________________________________  
    if(imputation_method!='KNN'):
        # MICE IMPUTATION
        mice_imputer = IterativeImputer()
        data.iloc[:, :] = mice_imputer.fit_transform(data)

    else:  
        # KNN IMPUTATION
        knn_imputer = KNN()
        data.iloc[:, :] = knn_imputer.fit_transform(data)
 
    print(data.shape)
    #_______________________________________Dim Reduction_______________________________________
    if(dim_reduction):
        # ANALYSIS OF VARIABLES AND THEIR CORRELATION 
        sns.set(rc={'figure.figsize':(40,40)})
        cmap=sns.diverging_palette(h_neg=10, h_pos=240, as_cmap=True)
        mask = np.triu(np.ones_like(data.corr().abs(), dtype=bool))
        dataset_corr = sns.heatmap(data.corr().abs(), mask=mask, center=0, cmap=cmap, linewidths=1, annot=True, fmt='.2f')
        dataset_corr.get_figure().savefig('check_me_out/column_correlation.png')
        
        drop_candidates = ['Exterior2nd', 'GarageCond', 'GarageCars', 'FireplaceQu']
        data = data.drop(columns=drop_candidates)
    
    return data

def run_pca(data, continuous_columns):
    #_______________________________________PCA_______________________________________
    
    pca_subset_df = data[continuous_columns]
    
    # NORMALIZE Data to improve PCA perform.
    df_std = (pca_subset_df - pca_subset_df.mean())/ pca_subset_df.std()
    pca = PCA(n_components = df_std.shape[1])  # 0.9.    or medical_cont_df.shape[1]
    pca.fit_transform(df_std)
    
    columns_gen_2 = ['PC'+ str(x) for x in range(1,len(df_std.columns)+1)]
    print(pca.explained_variance_ratio_.cumsum())
    loadings = pd.DataFrame( pca.components_.T, columns = columns_gen_2, index = pca_subset_df.columns )
    print(loadings)
    return loadings


#_______________________________________OUTPUT_______________________________________

train_knn_df = wrangle_data(original_train_df, dim_reduction=False)
pca = run_pca(train_knn_df, continuous_columns)
train_knn_df.to_csv('wrangled_data/train_knn_df.csv', index=False, )


train_knn_reduced_df = wrangle_data(original_train_df)
train_knn_reduced_df.to_csv('wrangled_data/train_knn_reduced_df.csv', index=False,)


train_mice_df = wrangle_data(original_train_df, dim_reduction=False, imputation_method='MICE')
train_mice_df.to_csv('wrangled_data/train_mice_df.csv', index=False,)


train_mice_reduced_df = wrangle_data(original_train_df, imputation_method='MICE')
train_mice_df.to_csv('wrangled_data/train_mice_reduced_df.csv', index=False,)

test_prepped_df = wrangle_data(test_df, is_test=True, dim_reduction=False)
test_prepped_df.to_csv('wrangled_data/test_prepped_df.csv', index=False)

