import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ---
INPUT_X_PATH = 'processed_X.csv'
INPUT_Y_PATH = 'processed_y.csv'
OUTPUT_MODEL_PATH = 'baseline_model.joblib'
OUTPUT_CM_PLOT_PATH = 'plots/confusion_matrix.png'

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load the preprocessed data
    try:
        X = pd.read_csv(INPUT_X_PATH)
        y = pd.read_csv(INPUT_Y_PATH).squeeze() # Use .squeeze() to make it a Series
        print("Successfully loaded processed data.")
    except FileNotFoundError:
        print("Error: Processed data files not found. Please run the previous scripts first.")
        exit()

    # 2. Split data into training and testing sets
    # `stratify=y` ensures the proportion of labels is the same in train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Data split into 80% training and 20% testing sets.")

    # 3. Train the baseline model (Logistic Regression)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print("Baseline model trained successfully.")

    # 4. Evaluate the model's performance on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Label 0 (Neutral)', 'Label 1 (Stressed)'])
    
    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy on Test Set: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("--------------------------------\n")
    
    # 5. Visualize and save the Confusion Matrix for better insight
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Neutral', 'Stressed'], yticklabels=['Neutral', 'Stressed'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(OUTPUT_CM_PLOT_PATH)
    print(f"Confusion matrix plot saved to '{OUTPUT_CM_PLOT_PATH}'.")

    # 6. Save the final trained model for future use
    joblib.dump(model, OUTPUT_MODEL_PATH)
    print(f"Trained model saved to '{OUTPUT_MODEL_PATH}'.")
