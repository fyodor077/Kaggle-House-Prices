# == Imports ==
import numpy as np
import pandas as pd

import xgboost as xgb
import lightgbm as lgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import warnings
warnings.filterwarnings('ignore')

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# == Config ==
CONFIG = {
    # -- Paths --
    'train_path': 'data/train.csv',
    'test_path': 'data/test.csv',

    # -- Target -- 
    'target': 'SalePriceLog',

    # -- Cross-validation (CV) --
    'n_folds': 5,
    'seed': 1337,

    # -- Features --
    'cat_features': [
        'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
        'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'Condition2',
        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtFinType1',
        'BsmtFinType2', 'BsmtExposure', 'Heating', 'CentralAir',
        'Electrical', 'Functional', 'GarageType', 'GarageFinish',
        'PavedDrive', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition',
        'Neighborhood'
        ],
    'drop_features': ['Id', 'Neighborhood'],

    # -- Model params --
    'ridge_params': {'alpha': 10.0,},
    'xgb_params': {
        'n_estimators': 500,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 1337,
        'verbosity': 0,
    },
    'lgb_params': {
        'n_estimators': 500,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 1337,
        'verbosity': -1,
    }
}

# == Load data ==
def load_data(config):
    train = pd.read_csv(config['train_path'])
    test = pd.read_csv(config['test_path'])
    test_ids = test['Id'].copy()
    return train, test, test_ids

train, test, test_ids = load_data(CONFIG)

# == Feature Engineering ==
def engineer_features(df, is_train=True, fit_params=None):
    df = df.copy()

    if is_train:
        fit_params={}
    
    # -- Target --
    if is_train:
        df['SalePriceLog'] = np.log1p(df['SalePrice'])
        df = df.drop(columns=['SalePrice'])
    
    # -- Missing Values --
    cat_none_cols = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
        'GarageType', 'GarageQual', 'GarageFinish', 'GarageCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'MasVnrType'
    ]
    for col in cat_none_cols:
        df[col] = df[col].fillna('None')
    
    num_zero_cols = [
        'GarageYrBlt', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
        'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
        'GarageCars', 'GarageArea'
    ]
    for col in num_zero_cols:
        df[col]=df[col].fillna(0)

    # -- LotFrontage --
    if is_train:
        fit_params['lot_median'] = df.groupby('Neighborhood')['LotFrontage'].median()
    df['LotFrontage'] = df['LotFrontage'].fillna(
        df['Neighborhood'].map(fit_params['lot_median'])
    )

    # -- Electrical --
    if is_train:
        fit_params['electrical_mode'] = df['Electrical'].mode()[0]
        fit_params['test_mode_cols'] = {
            col: df[col].mode()[0]
            for col in ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd',
                        'KitchenQual', 'Functional', 'SaleType']
        }
    df['Electrical'] = df['Electrical'].fillna(fit_params['electrical_mode'])
    for col, mode_val in fit_params['test_mode_cols'].items():
        if col in df.columns:
            df[col] = df[col].fillna(mode_val)
    
    # -- Outliers (train only) --
    if is_train:
        df = df[~((df['GrLivArea'] > 4000) & (df['SalePriceLog'] < 12.5))]
    
    # -- Ordinal Encoding --
    quality_mapping = {
        'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5
    }
    quality_cols = [
        'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
        'HeatingQC', 'KitchenQual', 'FireplaceQu',
        'GarageQual', 'GarageCond', 'PoolQC'
    ]
    for col in quality_cols:
        df[col] = df[col].map(quality_mapping)
    
    # -- Target Encoding for Neighborhood --
    if is_train:
        fit_params['neighborhood_enc'] = df.groupby('Neighborhood')['SalePriceLog'].median()
    df['NeighborhoodEnc'] = df['Neighborhood'].map(fit_params['neighborhood_enc'])

    # -- New Features --
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBath'] = (
        df['FullBath'] + df['HalfBath'] * 0.5 +
        df['BsmtFullBath'] + df['BsmtHalfBath'] * 0.5
    )
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['YearsSinceRemod'] = df['YrSold'] - df['YearRemodAdd']
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)

    # -- Drop --
    df = df.drop(columns=CONFIG['drop_features'], errors='ignore')

    return df, fit_params

# == Apllying ==
train, fit_params = engineer_features(train, is_train=True)
test, _= engineer_features(test, is_train=False, fit_params=fit_params)


# print("Cols in train:", train.columns.tolist())


# -- Target / Features split --
X_train = train.drop(columns=[CONFIG['target']])
y_train = train[CONFIG['target']]
X_test = test.copy()

# print(f'X_train shape: {X_train.shape}')
# print(f'X_test shape: {X_test.shape}')
# print(f'y_train shape: {y_train.shape}')
# print(X_train.isnull().sum().sum(), 'NaN in X_train')
# print(X_test.isnull().sum().sum(), 'NaN in X_test')

# == Preprocessing ==
def build_preprocessor(cat_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ('ohe', OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False
            ), cat_features),
        ],
        remainder='passthrough'
    )
    return preprocessor

preprocessor = build_preprocessor(CONFIG['cat_features'])

# == Cross-Validation & Training ==
def train_model(model, X_train, y_train, X_test, preprocessor, config, verbose=True):
    kf = KFold(
        n_splits=config['n_folds'],
        shuffle=True,
        random_state=config['seed']
    )

    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        if verbose:
            print(f'Fold {fold + 1}/{config["n_folds"]}', end=' | ')
        
# -- Split --
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

# -- Preprocessing -- 
        fold_preprocessor = clone(preprocessor)
        X_fold_train = fold_preprocessor.fit_transform(X_fold_train)
        X_fold_val = fold_preprocessor.transform(X_fold_val)
        X_fold_test = fold_preprocessor.transform(X_test)

# -- Training --
        fold_model = clone(model)
        fold_model.fit(X_fold_train, y_fold_train)

# -- Validation --
        val_preds = fold_model.predict(X_fold_val)
        fold_score = np.sqrt(mean_squared_error(y_fold_val, val_preds))
        scores.append(fold_score)

        if verbose:
            print(f'RMSE: {fold_score:.4f}')
        
# -- OOF & Test predictions --
        oof_preds[val_idx] = val_preds
        test_preds += fold_model.predict(X_fold_test) / config['n_folds']
    
    if verbose:
        print(f'\nMean RMSE: {np.mean(scores):.4f} ± {np.std(scores):.4f}')
        print(f'OOF RMSE: {np.sqrt(mean_squared_error(y_train, oof_preds)):.4f}')
    return oof_preds, test_preds, scores
