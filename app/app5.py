import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler, PowerTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, fbeta_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
import joblib
from preprocessing import log1p_transform
from preprocessing import fill_offer_na, fill_phone_service_depndents, fill_internet_add_ons, add_on_features, stream_features, calculate_clv, refund_flag_transformer, drop_raw_columns, add_zip_cv
from preprocessing import fill_offer_transformer, fill_internet_transformer, add_addons_transformer, tenure_engineering_transformer, calculate_clv_transformer, refund_flag_transformer, extra_data_flag_transformer, stream_feats_transformer, add_zip_cv_transformer
import pickle

with open(r"C:\Users\Rewan\Downloads\ML Customer Churn Prediction\zip_cv_map.pkl", "rb") as file:
    zip_cv_map = pickle.load(file)

with open(r"C:\Users\Rewan\Downloads\ML Customer Churn Prediction\global_zip_mean.pkl", "rb") as file:
    global_zip_mean = pickle.load(file)
# Set page config for better appearance
st.set_page_config(page_title="Churn Prediction", layout="wide")

# Dark theme styling
st.markdown("""
<style>
    body {
        background-color: #111111;
        color: white;
    }
    .reportview-container, .main, .block-container {
        background-color: #111111;
        color: white;
    }
    .header-style {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
        color: white;
    }
    .metric-container {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: white;
    }
    .customer-details {
        background-color: #2a2a2a;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
        color: white;
    }
    .prediction-result {
        font-size: 18px;
        font-weight: bold;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

st.title("Telecom Customer Churn Prediction")

# Load and preprocess data
MODEL_PATHS = {
    "Random Forest": r"C:\Users\Rewan\Downloads\ML Customer Churn Prediction\rf_best_estimator_.joblib",
    "Decision Tree": r"C:\Users\Rewan\Downloads\ML Customer Churn Prediction\dt_best_estimator_.joblib",
    "Logistic Regression": r"C:\\Users\\Rewan\\Downloads\\logistic_search_best_estimator_.joblib"
}


def load_models(paths):
    """Load all models and cache them for performance."""
    loaded = {}
    for name, path in paths.items():
        loaded[name] = joblib.load(path)
    return loaded

models = load_models(MODEL_PATHS)
accuracies = {}
for name, model in models.items():
    if hasattr(model, 'best_score_'):
        # For GridSearchCV or RandomizedSearchCV
        accuracies[name] = model.best_score_
    elif hasattr(model, 'score'):
        # Placeholder: requires training data; set to None or compute separately
        accuracies[name] = None
    else:
        accuracies[name] = None
st.cache_data
with st.sidebar:
        st.title("Model Settings")
        model_choice = st.selectbox("Select classifier",
                                  ("Random Forest", "Decision Tree", "Logistic Regression"))

if model_choice == "Random Forest":
        selected_model = "Random Forest"
        model = models["Random Forest"]
elif model_choice == "Decision Tree":
        selected_model = "Decision Tree"
        model = models["Decision Tree"]
elif model_choice == "Logistic Regression":
        selected_model = "Logistic Regression"
        model = models["Logistic Regression"]
        


st.markdown('<div class="header-style">Single Customer Prediction</div>', unsafe_allow_html=True)

with st.form("customer_input_form"):
    st.markdown("### Enter Customer Data")
    Age = st.number_input("Age", min_value=18, max_value=100, value=30)
    nod = st.number_input("Number of Dependents", min_value=0, max_value=1000, value=0)
    Nof = st.number_input("Number of Referrals", min_value=0, max_value=1000, value=0)
    tenure = st.number_input("Tenure in Months", min_value=0, max_value=100000, value=12)
    Avg_month_GB = st.number_input("Average Monthly GB Download", min_value=0.0, max_value=100000., value=50.0)
    monthly_charge = st.number_input("Monthly Charge", min_value=0.0, max_value=100000., value=70.0)
    Total_charges = st.number_input("Total Charges", min_value=0.0, max_value=1000000., value=1000.0)
    Total_refunds = st.number_input("Total Refunds", min_value=0.0, max_value=1000000., value=0.0)
    Total_extra_data = st.number_input("Total Extra Data Charges", min_value=0.0, max_value=1000000., value=0.0)
    Total_Long_distance = st.number_input("Total Long Distance Charges", min_value=0.0, max_value=1000000., value=0.0)
    Total_revenue = st.number_input("Total Revenue", min_value=0.0, max_value=1000000., value=0.0)
    zip_code = st.text_input("Zip Code", value="90000")
    
    marital_status = st.selectbox("Marital Status", ['Yes', 'No'])
    Offer = st.selectbox("Offer", ['None', 'Offer A', 'Offer B', 'Offer C', 'Offer D', 'Offer E'])
    phone_service = st.selectbox("Phone Service", ['Yes', 'No'])
    Internet_service = st.selectbox("Internet Service", ['Yes', 'No'])
    internet_Type = st.selectbox("Internet Type", ['Cable','DSL', 'Fiber Optic', 'No Internet Service'])
    Online_security = st.selectbox("Online Security", ['Yes', 'No', 'No Internet Service'])
    Online_backup = st.selectbox("Online Backup", ['Yes', 'No', 'No Internet Service'])
    Device_protection = st.selectbox("Device Protection", ['Yes', 'No', 'No Internet Service'])
    Tech_support = st.selectbox("Premium Tech Support", ['Yes', 'No', 'No Internet Service'])
    Unlimited_data = st.selectbox("Unlimited Data", ['Yes', 'No', 'No Internet Service'])
    Streaming_TV = st.selectbox("Streaming TV", ['Yes', 'No', 'No Internet Service'])
    
    Streaming_movies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No Internet Service'])
    Streaming_music = st.selectbox("Streaming Music", ['Yes', 'No', 'No Internet Service'])
    contract = st.selectbox("Contract Type", ['Month-to-Month', 'One Year', 'Two Year'])
    paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])
    Payment_method = st.selectbox("Payment Method", ['Bank Withdrawal', 'Credit Card', 'Mailed Check'])

    submitted = st.form_submit_button("Predict Churn")


# ——————————————————————————————
# 3) On submit: build DataFrame, predict, display
# ——————————————————————————————
if submitted:
    # assemble into dict
    input_dict = {
        'Age': Age,
        'Number of Dependents': nod,
        'Number of Referrals': Nof,
        'Tenure in Months': tenure,
        'Avg Monthly GB Download': Avg_month_GB,
        'Monthly Charge': monthly_charge,
        'Total Charges': Total_charges,
        'Total Refunds': Total_refunds,
        'Total Extra Data Charges': Total_extra_data,
        'Total Long Distance Charges': Total_Long_distance,
        'Total Revenue': Total_revenue,
        'Zip Code': zip_code,
        'Married': marital_status,
        'Offer': Offer,
        'Phone Service': phone_service,
        'Internet Service' : Internet_service,
        'Internet Type': internet_Type,
        'Online Security': Online_security,
        'Online Backup': Online_backup,
        'Device Protection Plan': Device_protection,
        'Premium Tech Support': Tech_support,
        'Unlimited Data' : Unlimited_data,
        'Streaming TV': Streaming_TV,
        'Streaming Movies': Streaming_movies,
        'Streaming Music': Streaming_music,
        'Contract': contract,
        'Paperless Billing': paperless_billing,
        'Payment Method': Payment_method
    }

    # create a one-row DataFrame
    # 1) Build your one‐row df from the Streamlit form
    df = pd.DataFrame([input_dict])
    
    
    st.write("**Input Summary**")
    st.dataframe(df)

    # get the selected model
    

    # run prediction
    try:
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0][1] if hasattr(model, 'predict_proba') else None

        # display
        st.success(f"**{selected_model}** predicts: **{'Churn' if pred==1 else 'No Churn'}**")
        if proba is not None:
            st.info(f"Predicted Churn Probability: **{proba*100:.2f}%**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")