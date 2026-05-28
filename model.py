
import joblib
import pandas as pd
import os

# Define the directory where models are saved
MODEL_DIR = 'models'

def load_artifacts():
    """Loads the pre-trained models, scaler, and feature columns."""
    try:
        linear_reg_model = joblib.load(os.path.join(MODEL_DIR, 'linear_reg_model.pkl'))
        random_forest_model = joblib.load(os.path.join(MODEL_DIR, 'random_forest_model.pkl'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        feature_columns = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))
        return linear_reg_model, random_forest_model, scaler, feature_columns
    except FileNotFoundError as e:
        print(f"Error loading model artifacts: {e}")
        print(f"Please ensure '{MODEL_DIR}' directory exists and contains all required .pkl files.")
        return None, None, None, None

def preprocess_input(input_data: pd.DataFrame, feature_columns, scaler):
    """Preprocesses new input data for prediction."""
    # Ensure consistent dummy variable creation
    # Note: 'sex', 'smoker', 'region' are the original categorical columns
    # The dummy variables created during training were 'sex_male', 'smoker_yes', 'region_northwest', etc.

    processed_input = input_data.copy()

    # Manually create dummy variables for consistency with training
    processed_input['sex_male'] = (processed_input['sex'] == 'male').astype(int)
    processed_input['smoker_yes'] = (processed_input['smoker'] == 'yes').astype(int)
    
    # Handle regions, assuming 'southeast' is the dropped one if drop_first=True was used
    processed_input['region_northwest'] = (processed_input['region'] == 'northwest').astype(int)
    processed_input['region_southeast'] = (processed_input['region'] == 'southeast').astype(int)
    processed_input['region_southwest'] = (processed_input['region'] == 'southwest').astype(int)

    # Drop original categorical columns
    processed_input = processed_input.drop(columns=['sex', 'smoker', 'region'])

    # Ensure all feature columns from training are present, add missing ones with 0
    for col in feature_columns:
        if col not in processed_input.columns:
            processed_input[col] = 0
            
    # Reorder columns to match the training data's feature order
    processed_input = processed_input[feature_columns]

    # Scale the numerical features
    scaled_input = scaler.transform(processed_input)
    return scaled_input

def predict_charges(model, preprocessed_input):
    """Makes a prediction using the given model and preprocessed input."""
    return model.predict(preprocessed_input)


if __name__ == '__main__':
    # Example usage (for testing model.py independently)
    linear_reg, random_forest, data_scaler, f_cols = load_artifacts()

    if linear_reg and random_forest and data_scaler and f_cols:
        print("Models and scaler loaded successfully.")

        # Example new data point
        sample_data = pd.DataFrame({
            'age': [30],
            'sex': ['female'],
            'bmi': [25.0],
            'children': [1],
            'smoker': ['no'],
            'region': ['northeast']
        })

        print("\nSample Input Data:")
        print(sample_data)

        preprocessed_sample = preprocess_input(sample_data, f_cols, data_scaler)
        print("\nPreprocessed and Scaled Input (first 5 values):", preprocessed_sample[0][:5])

        lr_prediction = predict_charges(linear_reg, preprocessed_sample)
        rf_prediction = predict_charges(random_forest, preprocessed_sample)

        print(f"\nLinear Regression Prediction: ${lr_prediction[0]:,.2f}")
        print(f"Random Forest Prediction: ${rf_prediction[0]:,.2f}")
    else:
        print("Failed to load artifacts. Cannot run example usage.")
