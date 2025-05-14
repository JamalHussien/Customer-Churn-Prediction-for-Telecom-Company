import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler, PowerTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
import pickle
from sklearn.base import BaseEstimator, TransformerMixin


with open(r"C:\Users\Rewan\Downloads\ML Customer Churn Prediction\zip_cv_map.pkl", "rb") as file:
    zip_cv_map = pickle.load(file)

with open(r"C:\Users\Rewan\Downloads\ML Customer Churn Prediction\global_zip_mean.pkl", "rb") as file:
    global_zip_mean = pickle.load(file)

def fill_offer_na(df):
    df['Offer'] = df['Offer'].fillna('No Offer')
    return df

def fill_phone_service_depndents(df):
    df['Avg Monthly Long Distance Charges'] = df['Avg Monthly Long Distance Charges'].fillna(0)
    # For categorical column
    df['Multiple Lines'] = df['Multiple Lines'].fillna('No Phone Service')
    return df

def fill_internet_add_ons(df):
    # For categorical features
    internet_cat_cols = ['Internet Type'] + ['Online Security','Online Backup',
                       'Device Protection Plan','Premium Tech Support',
                       'Streaming TV','Streaming Movies','Streaming Music',
                       'Unlimited Data']
    df[internet_cat_cols] = df[internet_cat_cols].fillna('No Internet Service')
    # For numeric feature
    df['Avg Monthly GB Download'] = df['Avg Monthly GB Download'].fillna(0)
    return df

def add_on_features(df, addons):
    df['AddOnCount'] = df[addons].apply(lambda row: sum(row == 'Yes'), axis=1)
    return df

addons = [
    'Online Security', 'Online Backup',
    'Device Protection Plan', 'Premium Tech Support',
]

def stream_features(df):
    df['StreamCount'] = df[['Streaming TV', 'Streaming Movies', 'Streaming Music']].apply(lambda row: sum(row == 'Yes'), axis=1)
    return df

def calculate_clv(df):
    """Calculate predictive CLV based on months remaining in the current contract cycle."""
    # Define contract durations
    contract_length_map = {
        'Month-to-Month': 1,
        'One Year': 12,
        'Two Year': 24
    }

    # Map to numerical contract lengths
    df['ContractLengthNum'] = df['Contract'].map(contract_length_map)

    # Calculate months remaining in current contract cycle
    df['RemainingMonths'] = df['ContractLengthNum'] - (df['Tenure in Months'] % df['ContractLengthNum'])

    # Month-to-month customers have no 'remaining' contract cycle
    df.loc[df['Contract'] == 'Month-to-month', 'RemainingMonths'] = 1

    # Final CLV prediction
    df['CLV'] = df['Monthly Charge'] * df['RemainingMonths']

    return df
def create_refund_flags(df):
  """Creates flags for refunds"""
  df['HadRefunds'] = (df['Total Refunds'] > 0).astype(int)
  return df

def extra_data_charge_flag(df):
    """Creates flags for extra data charges."""
    df['HadExtraDataCharges'] = (df['Total Extra Data Charges'] > 0).astype(int)
    return df

def cv_target_encode(df: pd.DataFrame,group_col: str,target_col: str,n_splits: int = 5,
    random_state: int = 42, suffix: str = '_CV') -> pd.DataFrame:
    """
    Perform cross-validated target mean encoding on `group_col` with respect to `target_col`.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    group_col : str
        The name of the categorical column to encode.
    target_col : str
        The binary target column name (e.g. 'Customer Status').
    n_splits : int, default=5
        Number of folds for KFold.
    random_state : int, default=42
        Random seed for shuffling.
    suffix : str, default='_CV'
        Suffix to append to `group_col` for the new encoded column.

    Returns
    -------
    pd.DataFrame
        The DataFrame with a new column: `<group_col><suffix>` containing the CV target means.
    """
    # Prepare output column
    new_col = f"{group_col}{suffix}"
    df[new_col] = np.nan

    # Global mean to fill unseen categories
    global_mean = df[target_col].mean()

    # Set up folds
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_pos, valid_pos in kf.split(df):
        # Convert positions to index labels
        train_idx = df.index[train_pos]
        valid_idx = df.index[valid_pos]

        # Compute mean target per category on the train split
        fold_means = (
            df.loc[train_idx]
              .groupby(group_col)[target_col]
              .mean()
        )

        # Map to the validation fold
        df.loc[valid_idx, new_col] = df.loc[valid_idx, group_col].map(fold_means)

    # Fill any missing (unseen categories) with global mean
    df[new_col].fillna(global_mean, inplace=True)

    return df

def tenure_and_contract_engineering(df):
    # Continuous + bucketed tenure
    df['TenureMonths'] = df['Tenure in Months']
    df['TenureBins'] = pd.cut(
        df['TenureMonths'],
        bins=[0, 6, 12, 24, 36, 60, 100],
        labels=['0–6','6–12','12–24','24–36','36–60','60+']
    )
    # Contract encoded ordinally
    contract_map = {'Month-to-Month': 0, 'One Year': 1, 'Two Year': 2}
    df['ContractLen'] = df['Contract'].map(contract_map)
    return df




def add_zip_cv(df):
    df = df.copy()
    df['ZipCode_CV'] = df['Zip Code'].map(zip_cv_map).fillna(global_zip_mean)
    return df

# zip_cv_transformer = FunctionTransformer(add_zip_cv, validate=False)


def fill_offer_transformer(df):
    return fill_offer_na(df.copy())
fill_offer_tf = FunctionTransformer(fill_offer_transformer, validate=False)
#def fill_phone_transformer(df):
#   return fill_phone_service_depndents(df.copy())

#fill_phone_tf = FunctionTransformer(fill_phone_transformer, validate=False)
def fill_internet_transformer(df):
    return fill_internet_add_ons(df.copy())

fill_internet_tf = FunctionTransformer(fill_internet_transformer, validate=False)

def add_addons_transformer(df):
    return add_on_features(df.copy(), addons)

add_addons_tf = FunctionTransformer(add_addons_transformer, validate=False)

def tenure_engineering_transformer(df):
    return tenure_and_contract_engineering(df.copy())

tenure_eng_tf = FunctionTransformer(tenure_engineering_transformer, validate=False)

def calculate_clv_transformer(df):
    return calculate_clv(df.copy())

calc_clv_tf = FunctionTransformer(calculate_clv_transformer, validate=False)

def refund_flag_transformer(df):
    return create_refund_flags(df.copy())

refund_flag_tf = FunctionTransformer(refund_flag_transformer, validate=False)

def extra_data_flag_transformer(df):
    return extra_data_charge_flag(df.copy())

extra_flag_tf = FunctionTransformer(extra_data_flag_transformer, validate=False)

def stream_feats_transformer(df):
    return stream_features(df.copy())

stream_feats_tf = FunctionTransformer(stream_feats_transformer, validate=False)
def add_zip_cv_transformer(df):
     return add_zip_cv(df.copy())

to_drop = ['Total Refunds', 'Total Extra Data Charges', 'Zip Code',  'Internet Service',
           'Online Security', 'Online Backup', 'Device Protection Plan',
             'Premium Tech Support', 'Streaming TV', 'Streaming Movies',
             'Streaming Music']
def log1p_transform(X):
    return np.log1p(X)

def drop_raw_columns(df):
    return df.drop(columns=to_drop)

drop_raw = FunctionTransformer(drop_raw_columns, validate=False)
