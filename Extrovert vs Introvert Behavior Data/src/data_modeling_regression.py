import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier


# Path to Directories
cur_dir = os.path.dirname(os.path.abspath(__file__))
csv_dir = os.path.join(cur_dir, '..', 'data')


# Regression imputation for numerical columns with validation
def regression_impute(df, target_col, features, verbose=True):
    df_temp = df[features + [target_col]].copy()

    complete_data = df_temp[df_temp[features + [target_col]].notnull().all(axis=1)]
    predict_data = df_temp[df_temp[target_col].isnull() & df_temp[features].notnull().all(axis=1)]

    # If no missing values to predict, return original
    if predict_data.empty:
        return df[target_col]

    # Evaluation phase: use complete cases to evaluate model
    X = complete_data[features]
    y = complete_data[target_col]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Validation
    if verbose:
        y_val_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_val_pred)
        print(f"[Eval] MSE for '{target_col}' regression imputation: {mse:.4f}")

    # Impute actual missing values
    X_pred = predict_data[features]
    y_pred = model.predict(X_pred)
    df.loc[predict_data.index, target_col] = y_pred

    return df[target_col]


# Logistic regression imputation for categorical columns
def logistic_regression_impute(df, target_col, features):
    df_temp = df[features + [target_col]].copy()
    train_data = df_temp[df_temp[target_col].notnull()]
    predict_data = df_temp[df_temp[target_col].isnull()]

    if predict_data.empty:
        return df[target_col]

    X_train = train_data[features]
    y_train = train_data[target_col]
    X_pred = predict_data[features]

    clf = HistGradientBoostingClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_pred)

    df.loc[predict_data.index, target_col] = y_pred
    return df[target_col]


if __name__ == "__main__":
    data = pd.read_csv(os.path.join(csv_dir, 'personality_dataset.csv'))

    # Define columns
    numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
                    'Friends_circle_size', 'Post_frequency']
    categorical_cols = ['Stage_fear', 'Drained_after_socializing']

    # Encode categorical temporarily
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        notnull_mask = data[col].notnull()
        data.loc[notnull_mask, col] = le.fit_transform(data.loc[notnull_mask, col])
        label_encoders[col] = le

    # Ensure float type for encoded categories
    data[categorical_cols] = data[categorical_cols].astype('float')

    # Impute numerical columns (with evaluation)
    for col in numeric_cols:
        predictors = [c for c in data.columns if c != col and c in numeric_cols + categorical_cols]
        data[col] = regression_impute(data, col, predictors)

    # Impute categorical columns
    for col in categorical_cols:
        predictors = [c for c in data.columns if c != col and c in numeric_cols]
        data[col] = logistic_regression_impute(data, col, predictors)
        # Convert back to original labels
        data[col] = label_encoders[col].inverse_transform(data[col].astype(int))

    # Final null check
    print("\nFinal null counts after imputation:\n")
    print(data.isnull().sum())