import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def convert_target_to_categorical(dataset, target_column):
    """
    Convert the target column to a categorical variable if it is continuous.
    If the target column is already categorical, this function does nothing.

    Args:
        dataset (pd.DataFrame): The dataset containing the target column.
        target_column (str): The name of the target column to convert.

    Returns:
        pd.DataFrame: The updated dataset with the target variable converted to categorical.
    """
    if pd.api.types.is_numeric_dtype(dataset[target_column]):
        # Define bins and labels for categorical conversion
        bins = [0, 2, 4, 6, 8, 10]
        labels = [1, 2, 3, 4, 5]
        # Convert the target column to categorical
        dataset[target_column] = pd.cut(dataset[target_column], bins=bins, labels=labels, include_lowest=True)
    return dataset

def handle_missing_values(dataset, target_column):
    """
    Remove rows from the dataset that have NaN values in the target column.

    Args:
        dataset (pd.DataFrame): The dataset to clean.
        target_column (str): The name of the target column.

    Returns:
        pd.DataFrame: The cleaned dataset without rows containing NaN values in the target column.
    """
    # Drop rows with missing values in the target column
    dataset = dataset.dropna(subset=[target_column])
    return dataset

def prepare_data_dataset(dataset, target_column):
    """
    Prepare the dataset for analysis by converting the target column to categorical
    and handling NaN values.

    Args:
        dataset (pd.DataFrame): The dataset to prepare.
        target_column (str): The name of the target column to prepare.

    Returns:
        pd.DataFrame: The prepared dataset.
    """
    # Convert the target column to categorical if necessary
    dataset = convert_target_to_categorical(dataset, target_column)
    # Remove rows with missing values in the target column
    dataset = handle_missing_values(dataset, target_column)
    return dataset

def prepare_data_for_fairness(dataset, sensitive_cols, target_col):
    """
    Prepare the dataset for fairness evaluations by encoding the sensitive columns
    and binarizing the target column.

    Args:
        dataset (pd.DataFrame): The dataset to prepare.
        sensitive_cols (list): The list of sensitive columns to encode.
        target_col (str): The name of the target column.

    Returns:
        tuple: The feature matrix (X), target vector (y), and the processed dataset.
    """
    # Initialize a LabelEncoder for encoding sensitive columns
    label_encoder = LabelEncoder()
    for col in sensitive_cols:
        # Encode each sensitive column
        dataset[col] = label_encoder.fit_transform(dataset[col])

    # Binarize the target column based on its median value
    dataset[target_col] = (dataset[target_col] > dataset[target_col].median()).astype(int)

    # Extract features (X) and target (y)
    X = dataset[sensitive_cols]
    y = dataset[target_col]

    return X, y, dataset

def sample_dataset(dataset, fraction=0.1):
    """
    Sample a fraction of the dataset for analysis.

    Args:
        dataset (pd.DataFrame): The dataset to sample.
        fraction (float): The fraction of the dataset to sample.

    Returns:
        pd.DataFrame: The sampled dataset.
    """
    # Randomly sample a fraction of the dataset
    return dataset.sample(frac=fraction, random_state=42)

def prepare_data_model(dataset, target_column):
    """
    Prepare the dataset for modeling by encoding categorical columns and scaling numerical columns.

    Args:
        dataset (pd.DataFrame): The dataset to prepare.
        target_column (str): The name of the target column.

    Returns:
        pd.DataFrame: The prepared dataset.
    """
    # Encode categorical variables using LabelEncoder
    categorical_columns = dataset.select_dtypes(include=['object']).columns
    label_encoders = {col: LabelEncoder() for col in categorical_columns}
    for col, le in label_encoders.items():
        dataset[col] = le.fit_transform(dataset[col])

    # Scale numerical variables using StandardScaler
    numerical_columns = dataset.select_dtypes(include=['number']).columns
    scaler = StandardScaler()
    dataset[numerical_columns] = scaler.fit_transform(dataset[numerical_columns])

    return dataset

def prepare_data_for_model_optimization(dataset, target_column, protected_attribute):
    """
    Prepare the dataset for model optimization, ensuring the protected attribute is retained.

    Args:
        dataset (pd.DataFrame): The dataset to prepare.
        target_column (str): The name of the target column.
        protected_attribute (str): The name of the protected attribute.

    Returns:
        tuple: (X, y, dataset) where X is the feature matrix, y is the target vector,
               and dataset is the processed dataset.
    """
    dataset = dataset.copy()
    if target_column not in dataset.columns or protected_attribute not in dataset.columns:
        raise ValueError("Both target_column and protected_attribute must be in the dataset")

    # Encode categorical variables
    categorical_columns = dataset.select_dtypes(include=['object']).columns
    label_encoders = {col: LabelEncoder() for col in categorical_columns}
    for col, le in label_encoders.items():
        dataset[col] = le.fit_transform(dataset[col])

    # Convert target to binary if it has more than two unique values
    if dataset[target_column].nunique() > 2:
        dataset[target_column] = (dataset[target_column] > dataset[target_column].median()).astype(int)

    # Prepare feature matrix (X) and target vector (y)
    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]

    # Ensure the protected attribute is included in X
    if protected_attribute not in X.columns:
        X[protected_attribute] = dataset[protected_attribute]

    return X, y, dataset

def preprocess_protected_attribute(X_df, protected_attribute):
    """
    Ensure the protected attribute is in a valid format for ThresholdOptimizer.
    If the attribute is not binary, it will be converted to a binary format.

    Args:
        X_df (pd.DataFrame): The DataFrame containing features.
        protected_attribute (str): The name of the protected attribute column.

    Returns:
        pd.Series: The processed protected attribute.
    """
    if protected_attribute not in X_df.columns:
        raise ValueError(f"Protected attribute '{protected_attribute}' not found in dataset columns.")

    # Check if the protected attribute is categorical
    if pd.api.types.is_categorical_dtype(X_df[protected_attribute]):
        categories = X_df[protected_attribute].cat.categories
    else:
        categories = X_df[protected_attribute].unique()

    # Convert to binary if there are more than two categories
    if len(categories) > 2:
        print(f"Warning: Reducing {protected_attribute} to binary for simplicity.")
        X_df[protected_attribute] = pd.qcut(X_df[protected_attribute], q=2, labels=False, duplicates='drop')

    return X_df[protected_attribute]

