import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# --- Configuration ---
INPUT_CSV = 'features_labeled.csv'
OUTPUT_SCALER_PATH = 'scaler.joblib'
OUTPUT_X_PATH = 'processed_X.csv'
OUTPUT_Y_PATH = 'processed_y.csv'

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load the labeled feature data
    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"Successfully loaded '{INPUT_CSV}'.")
    except FileNotFoundError:
        print(f"Error: '{INPUT_CSV}' not found. Please run the data exploration script first.")
        exit()


    X = df.drop(['filename', 'label'], axis=1)
    y = df['label']
    print("Features (X) and target (y) have been separated.")
    print("Shape of feature set (X):", X.shape)

    # 3. Scale the features using StandardScaler
    # This standardizes features by removing the mean and scaling to unit variance.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("\nFeatures scaled successfully.")

    # 4. Save the processed data and the scaler object
    
    # Convert the scaled numpy array back to a DataFrame to save with column headers
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Save the processed features and labels to new CSV files
    X_scaled_df.to_csv(OUTPUT_X_PATH, index=False)
    y.to_csv(OUTPUT_Y_PATH, index=False)
    print(f"Processed features saved to '{OUTPUT_X_PATH}'")
    print(f"Labels saved to '{OUTPUT_Y_PATH}'")

    joblib.dump(scaler, OUTPUT_SCALER_PATH)
    print(f"Scaler object saved to '{OUTPUT_SCALER_PATH}'.")
    