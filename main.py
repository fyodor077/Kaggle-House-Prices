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
from sklearn.compose import make_column_selector
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
        'PavedDrive', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'
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

# -- Target / Features split --
X_train = train.drop(columns=[CONFIG['target']])
y_train = train[CONFIG['target']]
X_test = test.copy()

# == Preprocessing ==
def build_preprocessor(cat_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), make_column_selector(dtype_include=np.number)),
            ('cat', OneHotEncoder(
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

# == Hyperparameter Tuning ==
def xgb_objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'random_state': CONFIG['seed'],
        'verbosity': 0,
    }
    model = xgb.XGBRegressor(**params)
    _, _, scores = train_model(
        model, X_train, y_train, X_test, preprocessor, CONFIG, verbose=False
    )
    return np.mean(scores)

def lgb_objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'random_state': CONFIG['seed'],
        'verbosity': -1,
    }
    model = lgb.LGBMRegressor(**params)
    _, _, scores = train_model(
        model, X_train, y_train, X_test, preprocessor, CONFIG, verbose=False
    )
    return np.mean(scores)

# -- Run Tuning --
N_TRIALS = 50

print('Tuning XGBoost...')
xgb_study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=CONFIG['seed'])
)
xgb_study.optimize(xgb_objective, n_trials=N_TRIALS, show_progress_bar=True)
print(f'XGBoost best RMSE: {xgb_study.best_value:.4f}')
print(f'XGBoost best params: {xgb_study.best_params}')

print('Tuning LightGBM...')
lgb_study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=CONFIG['seed'])
)
lgb_study.optimize(lgb_objective, n_trials=N_TRIALS, show_progress_bar=True)
print(f'LightGBM best RMSE: {lgb_study.best_value:.4f}')
print(f'LightGBM best params: {lgb_study.best_params}')

# -- Update CONFIG with best params --
CONFIG['xgb_params'].update(xgb_study.best_params)
CONFIG['lgb_params'].update(lgb_study.best_params)

# == Models ==
baseline = Ridge(**CONFIG['ridge_params'])
xgb_model = xgb.XGBRegressor(**CONFIG['xgb_params'])
lgb_model = lgb.LGBMRegressor(**CONFIG['lgb_params'])

models = {
    'Baseline (Ridge)': baseline,
    'XGBoost': xgb_model,
    'LightGBM': lgb_model,
}

# == Run CV & Training ==
results = {}

for name, model in models.items():
    print(f'\n{"="*40}')
    print(f'Model: {name}')
    print(f'{"="*40}')

    oof_preds, test_preds, scores = train_model(
        model = model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        preprocessor=preprocessor,
        config=CONFIG
    )
    results[name] = {
        'oof_preds': oof_preds,
        'test_preds': test_preds,
        'scores': scores,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
    }

# == Results Summary ==
print(f'\n{"="*40}')
print('Results Summary')
print(f'{"="*40}')
for name, result in results.items():
    print(f'{name:25s} | RMSE: {result["mean_score"]:.4f} ± {result["std_score"]:.4f}')

# == MLP Model ==
# -- Network architecture --
class HouseMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.05),

            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.network(x)
    
def train_mlp(X_train, y_train, X_test, preprocessor, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    kf = KFold(
        n_splits=config['n_folds'],
        shuffle=True,
        random_state=config['seed']
    )

    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f'Fold {fold +1}/{config["n_folds"]}', end=' | ')

# -- Split --
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx].values
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx].values

# -- Preprocessing --
        fold_preprocessor = clone(preprocessor)
        X_fold_train_ohe = fold_preprocessor.fit_transform(X_fold_train)
        X_fold_val_ohe = fold_preprocessor.transform(X_fold_val)
        X_fold_test_ohe = fold_preprocessor.transform(X_test)

# -- Target Scaling --
        y_scaler = StandardScaler()
        y_fold_train_sc = y_scaler.fit_transform(y_fold_train.reshape(-1, 1))
        y_fold_val_sc = y_scaler.transform(y_fold_val.reshape(-1, 1))

# -- Tensors --
        X_tr = torch.FloatTensor(X_fold_train_ohe).to(device)
        y_tr = torch.FloatTensor(y_fold_train_sc).to(device)
        X_vl = torch.FloatTensor(X_fold_val_ohe).to(device)
        y_vl = torch.FloatTensor(y_fold_val_sc).to(device)
        X_te = torch.FloatTensor(X_fold_test_ohe).to(device)

        train_dataset = TensorDataset(X_tr, y_tr)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# -- Model --
        input_dim = X_fold_train_ohe.shape[1]
        model = HouseMLP(input_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# -- Early stopping --
        best_val_rmse = np.inf
        best_model_weights = None
        patience = 20
        patience_counter = 0
        max_epochs = 300

        for epoch in range(max_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_preds = model(X_vl)
                val_rmse = torch.sqrt(criterion(val_preds, y_vl)).item()
            
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_model_weights = {
                    k: v.clone() for k, v in model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
# -- Make predictions with best weights --
        model.load_state_dict(best_model_weights)
        model.eval()
        with torch.no_grad():
            val_preds_sc = model(X_vl).cpu().numpy()
            test_preds_sc = model(X_te).cpu().numpy()

            val_final = y_scaler.inverse_transform(val_preds_sc).squeeze()
            test_final = y_scaler.inverse_transform(test_preds_sc).squeeze()
        
        fold_score = np.sqrt(mean_squared_error(y_fold_val, val_final))
        scores.append(fold_score)
        print(f'RMSE: {fold_score:.4f} | Stopped at epoch: {epoch + 1}')

        oof_preds[val_idx] = val_final
        test_preds += test_final / config['n_folds']
    print(f'\nMean RMSE: {np.mean(scores):.4f} ± {np.std(scores):.4f}')
    print(f'OOF RMSE:  {np.sqrt(mean_squared_error(y_train, oof_preds)):.4f}')

    return oof_preds, test_preds, scores
    
# -- Launch MLP --
print(f'\n{"="*40}')
print('Model: MLP (PyTorch)')
print(f'{"="*40}')

mlp_oof, mlp_test_preds, mlp_scores = train_mlp(
    X_train, y_train, X_test, preprocessor, CONFIG
)

results['MLP (PyTorch)'] = {
    'oof_preds':  mlp_oof,
    'test_preds': mlp_test_preds,
    'scores':     mlp_scores,
    'mean_score': np.mean(mlp_scores),
    'std_score':  np.std(mlp_scores),
}

# == Ensemble Blending with Optuna ==
print(f'\n{"="*40}')
print('Tuning Ensemble Weights with Optuna...')
print(f'\n{"="*40}')

def blend_objective(trial):
    w_ridge = trial.suggest_float('w_Ridge', 0.0, 1.0)
    w_xgb = trial.suggest_float('w_XGBoost', 0.0, 1.0)
    w_lgb = trial.suggest_float('w_LightGBM', 0.0, 1.0)
    w_mlp = trial.suggest_float('w_MLP', 0.0, 1.0)

    total_weights = w_ridge + w_xgb + w_lgb + w_mlp
    if total_weights == 0:
        return np.inf
    
    w_ridge /= total_weights
    w_xgb /= total_weights
    w_lgb /= total_weights
    w_mlp /= total_weights

    blend_preds = (
        w_ridge * results['Baseline (Ridge)']['oof_preds'] +
        w_xgb * results['XGBoost']['oof_preds'] +
        w_lgb * results['LightGBM']['oof_preds'] +
        w_mlp * results['MLP (PyTorch)']['oof_preds']
    )
    return np.sqrt(mean_squared_error(y_train, blend_preds))

blend_study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=CONFIG['seed'])
)
blend_study.optimize(blend_objective, n_trials=100, show_progress_bar=True)

best_raw = blend_study.best_params
total_best_weight = sum(best_raw.values())

best_weights = {
    'Baseline (Ridge)': best_raw['w_Ridge'] / total_best_weight,
    'XGBoost': best_raw['w_XGBoost'] / total_best_weight,
    'LightGBM': best_raw['w_LightGBM'] / total_best_weight,
    'MLP (PyTorch)': best_raw['w_MLP'] / total_best_weight
}

print("\nOptimal Weights found by Optuna:")
for name, w in best_weights.items():
    print(f"{name:20s} : {w:.4f}")

blend_oof = np.zeros(len(y_train))
blend_test = np.zeros(len(test_ids))

for name, weight in best_weights.items():
    blend_oof += weight * results[name]['oof_preds']
    blend_test += weight * results[name]['test_preds']

blend_rmse = np.sqrt(mean_squared_error(y_train, blend_oof))
print(f'\nOptimized Ensemble OOF RMSE: {blend_rmse:.4f}')

results['Ensemble (Blend)'] = {
    'oof_preds':  blend_oof,
    'test_preds': blend_test,
    'scores':     [blend_rmse],
    'mean_score': blend_rmse,
    'std_score':  0.0,
}


# == Submission ==
def make_submission(results, test_ids, config):
    best_model_name = min(results, key=lambda x: results[x]['mean_score'])
    best_test_preds = results[best_model_name]['test_preds']

    print(f'\nBest model: {best_model_name}')
    print(f'Best CV RMSE: {results[best_model_name]["mean_score"]:.4f}')

    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': np.expm1(best_test_preds)
    })

    submission.to_csv('submission.csv', index=False)
    print(f'\nSubmission saved: submission.csv')
    print(submission['SalePrice'].describe())

make_submission(results, test_ids, CONFIG)