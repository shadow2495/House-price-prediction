import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.stats import skew, norm

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib

warnings.filterwarnings('ignore')
os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False
sns.set_palette("muted")


# 1. LOAD DATA

print("=" * 55)
print("STEP 1 — Loading data")
print("=" * 55)

train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test  shape: {test.shape}")
print(f"\nTarget (SalePrice) stats:\n{train['SalePrice'].describe().round(0)}")


# 2. EDA

print("\n" + "=" * 55)
print("STEP 2 — Exploratory Data Analysis")
print("=" * 55)

# --- 2a. Target distribution ---
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(train['SalePrice'], bins=60, color='#378ADD', edgecolor='white')
axes[0].set_title('SalePrice Distribution (raw)')
axes[0].set_xlabel('Sale Price ($)')
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x/1000:.0f}K'))

axes[1].hist(np.log1p(train['SalePrice']), bins=60, color='#1D9E75', edgecolor='white')
axes[1].set_title('SalePrice Distribution (log-transformed)')
axes[1].set_xlabel('log(1 + SalePrice)')
plt.suptitle('Target Variable — Skew correction', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('plots/01_target_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

skewness = train['SalePrice'].skew()
print(f"SalePrice skewness (raw):  {skewness:.3f}")
print(f"SalePrice skewness (log):  {np.log1p(train['SalePrice']).skew():.3f}")

# --- 2b. Correlation heatmap (top 12 numeric features) ---
corr_matrix = train.select_dtypes(include='number').corr()
top_features = corr_matrix['SalePrice'].abs().sort_values(ascending=False).head(12).index

fig, ax = plt.subplots(figsize=(11, 8))
sns.heatmap(train[top_features].corr(), annot=True, fmt='.2f',
            cmap='Blues', ax=ax, linewidths=0.5,
            cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Heatmap — Top 12 Features vs SalePrice', fontsize=12)
plt.tight_layout()
plt.savefig('plots/02_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# --- 2c. Top numeric correlations bar chart ---
top_corr = corr_matrix['SalePrice'].drop('SalePrice').abs().sort_values(ascending=False).head(15)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(top_corr.index[::-1], top_corr.values[::-1], color='#185FA5', edgecolor='white')
ax.set_title('Top 15 Features Correlated with SalePrice', fontsize=12)
ax.set_xlabel('Absolute Correlation')
plt.tight_layout()
plt.savefig('plots/03_top_correlations.png', dpi=150, bbox_inches='tight')
plt.show()

# --- 2d. Key scatter plots ---
key_features = ['GrLivArea', 'TotalBsmtSF', 'GarageArea', 'OverallQual']
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, feat in zip(axes.flatten(), key_features):
    ax.scatter(train[feat], train['SalePrice'], alpha=0.4, s=15, color='#378ADD')
    ax.set_xlabel(feat)
    ax.set_ylabel('SalePrice ($)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x/1000:.0f}K'))
    ax.set_title(f'{feat} vs SalePrice')
plt.suptitle('Key Feature Scatter Plots', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('plots/04_scatter_plots.png', dpi=150, bbox_inches='tight')
plt.show()

# --- 2e. Missing values ---
missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(missing.index, missing.values, color='#D85A30', edgecolor='white')
ax.set_title('Top 20 Columns with Missing Values (Train)', fontsize=12)
ax.set_ylabel('Missing Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plots/05_missing_values.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nColumns with missing values: {(train.isnull().sum() > 0).sum()}")


# 3. FEATURE ENGINEERING

print("\n" + "=" * 55)
print("STEP 3 — Feature Engineering")
print("=" * 55)

# Log-transform target
y = np.log1p(train['SalePrice'])
test_ids = test['Id']

# Combine train + test for consistent preprocessing
all_data = pd.concat([train.drop(['Id', 'SalePrice'], axis=1),
                      test.drop('Id', axis=1)], axis=0).reset_index(drop=True)

print(f"Combined data shape: {all_data.shape}")

# Fill missing values
# Categorical — NaN often means "None" for these features
none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
             'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
             'BsmtFinType2', 'MasVnrType', 'MSSubClass']
for col in none_cols:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna('None')

# Numeric — fill with 0 where NaN = no feature
zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars',
             'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
             'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
for col in zero_cols:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna(0)

# LotFrontage — fill by neighbourhood median
all_data['LotFrontage'] = (all_data.groupby('Neighborhood')['LotFrontage']
                                    .transform(lambda x: x.fillna(x.median())))

# Remaining categoricals — mode fill
cat_cols = all_data.select_dtypes(include='object').columns
all_data[cat_cols] = all_data[cat_cols].fillna(all_data[cat_cols].mode().iloc[0])

# Remaining numerics — median fill
num_cols = all_data.select_dtypes(include='number').columns
all_data[num_cols] = all_data[num_cols].fillna(all_data[num_cols].median())

# Create new features 
all_data['TotalSF']       = (all_data['TotalBsmtSF'] + all_data['1stFlrSF']
                              + all_data['2ndFlrSF'])
all_data['TotalBathrooms'] = (all_data['FullBath'] + 0.5 * all_data['HalfBath']
                               + all_data['BsmtFullBath'] + 0.5 * all_data['BsmtHalfBath'])
all_data['TotalPorchSF']  = (all_data['OpenPorchSF'] + all_data['EnclosedPorch']
                              + all_data['3SsnPorch'] + all_data['ScreenPorch'])
all_data['HasPool']       = (all_data['PoolArea'] > 0).astype(int)
all_data['HasGarage']     = (all_data['GarageArea'] > 0).astype(int)
all_data['HasBsmt']       = (all_data['TotalBsmtSF'] > 0).astype(int)
all_data['HasFireplace']  = (all_data['Fireplaces'] > 0).astype(int)
all_data['HouseAge']      = all_data['YrSold'] - all_data['YearBuilt']
all_data['RemodAge']      = all_data['YrSold'] - all_data['YearRemodAdd']
all_data['IsNew']         = (all_data['YearBuilt'] == all_data['YrSold']).astype(int)

print("New features created: TotalSF, TotalBathrooms, TotalPorchSF, "
      "HasPool, HasGarage, HasBsmt, HasFireplace, HouseAge, RemodAge, IsNew")

#Fix skewed numeric features
numeric_feats = all_data.select_dtypes(include='number').columns
skewed = all_data[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed = skewed[abs(skewed) > 0.75].index
all_data[skewed] = np.log1p(all_data[skewed])

print(f"Log-transformed {len(skewed)} skewed numeric features")

#  Label encode ordinal categoricals 
le = LabelEncoder()
cat_cols = all_data.select_dtypes(include='object').columns
for col in cat_cols:
    all_data[col] = le.fit_transform(all_data[col].astype(str))

print(f"Encoded {len(cat_cols)} categorical features")
print(f"\nFinal feature count: {all_data.shape[1]}")

# Split back\
n_train = len(train)
X       = all_data[:n_train].values
X_test  = all_data[n_train:].values

# Train / validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42)

# Scale
scaler  = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Val   set: {X_val.shape[0]} samples")


# 4. MODEL TRAINING + MLFLOW TRACKING

print("\n" + "=" * 55)
print("STEP 4 — Model Training with MLflow tracking")
print("=" * 55)

mlflow.set_experiment("house_price_prediction")

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate(model, X_tr, y_tr, X_v, y_v, name="model"):
    y_pred_val = model.predict(X_v)
    r2   = r2_score(y_v, y_pred_val)
    mae  = mean_absolute_error(y_v, y_pred_val)
    rms  = rmse(y_v, y_pred_val)
    # CV score
    cv   = cross_val_score(model, X_tr, y_tr,
                           cv=KFold(5, shuffle=True, random_state=42),
                           scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv.mean())
    print(f"\n{name:30s}  R²={r2:.4f}  RMSE={rms:.4f}  "
          f"MAE={mae:.4f}  CV-RMSE={cv_rmse:.4f}")
    return {'r2': r2, 'rmse': rms, 'mae': mae, 'cv_rmse': cv_rmse,
            'y_pred': y_pred_val}

models = {
    "Ridge (alpha=10)":
        Ridge(alpha=10),
    "Lasso (alpha=0.0005)":
        Lasso(alpha=0.0005, max_iter=3000),
    "ElasticNet":
        ElasticNet(alpha=0.0005, l1_ratio=0.9, max_iter=3000),
    "Random Forest":
        RandomForestRegressor(n_estimators=300, max_depth=15,
                              min_samples_leaf=2, random_state=42, n_jobs=-1),
    "Gradient Boosting":
        GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                  max_depth=4, subsample=0.8, random_state=42),
    "XGBoost":
        xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05,
                         max_depth=4, subsample=0.8,
                         colsample_bytree=0.8, reg_alpha=0.005,
                         random_state=42, n_jobs=-1,
                         early_stopping_rounds=50,
                         eval_metric='rmse'),
}

results = {}
trained_models = {}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        if name == "XGBoost":
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      verbose=False)
        else:
            model.fit(X_train, y_train)

        res = evaluate(model, X_train, y_train, X_val, y_val, name)
        results[name] = res
        trained_models[name] = model

        # Log to MLflow
        mlflow.log_param("model_type", name)
        mlflow.log_metric("r2",       res['r2'])
        mlflow.log_metric("rmse",     res['rmse'])
        mlflow.log_metric("mae",      res['mae'])
        mlflow.log_metric("cv_rmse",  res['cv_rmse'])


# 5. MODEL COMPARISON PLOT

print("\n" + "=" * 55)
print("STEP 5 — Model Comparison")
print("=" * 55)

names     = list(results.keys())
r2_scores = [results[n]['r2']      for n in names]
rmse_sc   = [results[n]['cv_rmse'] for n in names]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors_bar = ['#D85A30' if s == max(r2_scores) else '#378ADD' for s in r2_scores]

axes[0].barh(names, r2_scores, color=colors_bar, edgecolor='white')
axes[0].set_title('R² Score (higher = better)', fontsize=12)
axes[0].set_xlabel('R² Score')
axes[0].axvline(0.9, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
for i, v in enumerate(r2_scores):
    axes[0].text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=9)

colors_bar2 = ['#D85A30' if s == min(rmse_sc) else '#1D9E75' for s in rmse_sc]
axes[1].barh(names, rmse_sc, color=colors_bar2, edgecolor='white')
axes[1].set_title('CV RMSE (lower = better)', fontsize=12)
axes[1].set_xlabel('CV RMSE (log scale)')
for i, v in enumerate(rmse_sc):
    axes[1].text(v + 0.0005, i, f'{v:.4f}', va='center', fontsize=9)

plt.suptitle('Model Comparison', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('plots/06_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()


# 6. BEST MODEL — DETAILED ANALYSIS

best_name  = max(results, key=lambda n: results[n]['r2'])
best_model = trained_models[best_name]
best_res   = results[best_name]

print(f"\nBest model: {best_name}")
print(f"  R²      = {best_res['r2']:.4f}")
print(f"  RMSE    = {best_res['rmse']:.4f}")
print(f"  MAE     = {best_res['mae']:.4f}")
print(f"  CV-RMSE = {best_res['cv_rmse']:.4f}")

y_pred = best_res['y_pred']

# --- Actual vs Predicted ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(y_val, y_pred, alpha=0.5, s=15, color='#185FA5')
mn = min(y_val.min(), y_pred.min())
mx = max(y_val.max(), y_pred.max())
axes[0].plot([mn, mx], [mn, mx], 'r--', linewidth=1.2, label='Perfect fit')
axes[0].set_title(f'Actual vs Predicted — {best_name}')
axes[0].set_xlabel('Actual log(SalePrice)')
axes[0].set_ylabel('Predicted log(SalePrice)')
axes[0].legend()

residuals = y_val - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.4, s=15, color='#D85A30')
axes[1].axhline(0, color='black', linewidth=1.2, linestyle='--')
axes[1].set_title('Residuals Plot')
axes[1].set_xlabel('Predicted log(SalePrice)')
axes[1].set_ylabel('Residuals')

plt.suptitle(f'{best_name} — Prediction Analysis', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('plots/07_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Feature Importance (XGBoost or RF) ---
if best_name in ["XGBoost", "Random Forest", "Gradient Boosting"]:
    if hasattr(best_model, 'feature_importances_'):
        feat_names = all_data.columns.tolist()
        importances = best_model.feature_importances_
        fi_df = (pd.DataFrame({'Feature': feat_names, 'Importance': importances})
                   .sort_values('Importance', ascending=False)
                   .head(20))
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(fi_df['Feature'][::-1], fi_df['Importance'][::-1],
                color='#1D9E75', edgecolor='white')
        ax.set_title(f'Top 20 Feature Importances — {best_name}', fontsize=12)
        ax.set_xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('plots/08_feature_importance.png', dpi=150, bbox_inches='tight')
        plt.show()


# 7. ENSEMBLE (XGBoost + GBM + Ridge)


print("\n" + "=" * 55)
print("STEP 6 — Ensemble Prediction")
print("=" * 55)

xgb_pred  = trained_models["XGBoost"].predict(X_val)
gbm_pred  = trained_models["Gradient Boosting"].predict(X_val)
rdg_pred  = trained_models["Ridge (alpha=10)"].predict(X_val)
ens_pred  = 0.5 * xgb_pred + 0.3 * gbm_pred + 0.2 * rdg_pred

ens_r2   = r2_score(y_val, ens_pred)
ens_rmse = rmse(y_val, ens_pred)
print(f"Ensemble  R²={ens_r2:.4f}  RMSE={ens_rmse:.4f}")


# 8. SAVE BEST MODEL

joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(scaler,     'models/scaler.pkl')
joblib.dump(all_data.columns.tolist(), 'models/feature_names.pkl')

print("\nSaved: models/best_model.pkl, models/scaler.pkl, models/feature_names.pkl")
print("\nAll plots saved to plots/")
print("\nTraining complete!")
