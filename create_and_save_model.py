import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Main function
def create_and_save_model():
    # Path to the dataset
    file_path = input("Enter the path to the dataset (e.g., '/path/to/dataset.csv'): ").strip()

    try:
        # Loading the dataset
        dataset = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    # Specify the sensitive columns
    sensitive_cols = ["Sex_Code_Text"]

    # Encode the sensitive columns
    label_encoder = LabelEncoder()
    for col in sensitive_cols:
        dataset[col] = label_encoder.fit_transform(dataset[col])

    # Define feature and target columns
    feature_cols = sensitive_cols
    target_col = "DecileScore"

    # Convert the target variable to binary labels
    dataset[target_col] = (dataset[target_col] > 5).astype(int)

    # Split the dataset into train and test sets
    X = dataset[feature_cols]
    y = dataset[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict and calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Save the model
    output_dir = input("Enter the directory to save the model (e.g., '/path/to/save'): ").strip()
    os.makedirs(output_dir, exist_ok=True)
    initial_model_path = os.path.join(output_dir, 'initial_model.pkl')
    joblib.dump(model, initial_model_path)
    print(f"Initial model saved at: {initial_model_path}")

# Execute the main function
if __name__ == "__main__":
    create_and_save_model()
