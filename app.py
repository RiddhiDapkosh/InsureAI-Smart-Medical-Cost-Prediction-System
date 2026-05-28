
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor # Added for Random Forest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- Functions (copied or adapted from your notebook) ---

@st.cache_data # Cache data loading for performance
def load_data():
    """Loads the insurance data from 'insurance.csv'."""
    try:
        df = pd.read_csv("insurance.csv")
    except FileNotFoundError:
        st.error("Error: insurance.csv not found. Please ensure the file is in the same directory as app.py.")
        st.stop()
    return df

@st.cache_data # Cache preprocessing for performance
def preprocess_and_train_models(df):
    """Preprocesses the data, scales it, and trains both Linear Regression and Random Forest models."""
    # Create dummy variables for categorical features
    df_processed = pd.get_dummies(df.copy(), columns=['sex', 'smoker', 'region'], drop_first=True)

    # Define features (X) and target (y)
    X = df_processed.drop('charges', axis=1)
    y = df_processed['charges']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # We don't necessarily need to transform X_test_scaled here for the app, but good practice to keep it consistent

    # Train Linear Regression model
    linear_reg_model = LinearRegression()
    linear_reg_model.fit(X_train_scaled, y_train)

    # Train Random Forest Regressor model
    random_forest_model = RandomForestRegressor(random_state=42)
    random_forest_model.fit(X_train_scaled, y_train)

    return linear_reg_model, random_forest_model, scaler, X.columns

# --- Streamlit App Layout ---

st.set_page_config(page_title="Insurance Charges Predictor", layout="centered")
st.title("💸 Insurance Charges Prediction App")
st.markdown("This app predicts medical insurance charges based on various health parameters using different machine learning models.")

# Load data and train models
st.sidebar.header("Data Loading & Model Training")
with st.spinner("Loading data and training models..."):
    data = load_data()
    linear_reg_model, random_forest_model, scaler, feature_columns = preprocess_and_train_models(data)
st.sidebar.success("Data loaded and models trained!")

# Model Selection
selected_model_name = st.sidebar.selectbox(
    "Choose a Prediction Model",
    ("Linear Regression", "Random Forest Regressor")
)

if selected_model_name == "Linear Regression":
    model = linear_reg_model
    st.sidebar.caption("Selected: Linear Regression")
else:
    model = random_forest_model
    st.sidebar.caption("Selected: Random Forest Regressor")

st.header("Input Your Health Parameters")
st.write("Adjust the sliders and select boxes below to get a prediction.")

# Create input widgets for prediction
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 65, 30, help="Your age in years.")
        sex = st.selectbox("Sex", ["female", "male"], help="Your biological sex.")
        children = st.slider("Number of Children", 0, 5, 0, help="The number of dependents you have.")
    with col2:
        bmi = st.number_input("BMI (Body Mass Index)", 15.0, 50.0, 25.0, step=0.1, format="%.1f", help="Your BMI.")
        smoker = st.selectbox("Smoker", ["no", "yes"], help="Do you smoke?")
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"], help="Your residential area in the US.")

    st.markdown("--- Magnifier ---")
    submitted = st.form_submit_button("Predict Insurance Charges")

    if submitted:
        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'age': [age],
            'bmi': [bmi],
            'children': [children]
        })

        # Manually create dummy variables for input matching training data
        # Ensure consistent column names as in preprocessing
        input_data['sex_male'] = 1 if sex == 'male' else 0
        input_data['smoker_yes'] = 1 if smoker == 'yes' else 0
        input_data['region_northwest'] = 1 if region == 'northwest' else 0
        input_data['region_southeast'] = 1 if region == 'southeast' else 0
        input_data['region_southwest'] = 1 if region == 'southwest' else 0

        # Ensure all feature columns used in training are present, and in the correct order
        for col in feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0 # Add missing dummy columns with 0

        input_data = input_data[feature_columns] # Reorder to match model's expected input order

        # Scale the input data using the *fitted* scaler
        input_data_scaled = scaler.transform(input_data)

        prediction = model.predict(input_data_scaled)[0]
        st.success(f"#### Predicted Annual Insurance Charges: **${prediction:,.2f}**")
        st.balloons()

st.sidebar.header("Model Details")
if selected_model_name == "Linear Regression":
    st.sidebar.subheader("Linear Regression Coefficients")
    coefficients_df = pd.DataFrame({'Feature': feature_columns, 'Coefficient': linear_reg_model.coef_})
    st.sidebar.write(coefficients_df)
else:
    st.sidebar.subheader("Random Forest Information")
    st.sidebar.write(f"Number of Estimators: {random_forest_model.n_estimators}")
    st.sidebar.write(f"Max Depth: {random_forest_model.max_depth}") # Example attribute

st.sidebar.caption("Data source: Kaggle - Medical Cost Personal Datasets")
