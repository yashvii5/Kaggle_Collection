import os
import pandas as pd
# import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# Now impute categorical columns with logistic regression
from sklearn.linear_model import LogisticRegression

# Path to Directories
cur_dir = os.path.dirname(os.path.abspath(__file__))
csv_dir = os.path.join(cur_dir, '..', 'data')


# Regression imputation for numerical columns
def regression_impute(df, target_col, features):
    df_temp = df[features + [target_col]].copy()
    train_data = df_temp[df_temp[target_col].notnull()]
    predict_data = df_temp[df_temp[target_col].isnull()]

    if predict_data.empty:
        return df[target_col]

    x_train = train_data[features]
    y_train = train_data[target_col]
    x_pred = predict_data[features]

    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_pred)

    df.loc[df[target_col].isnull(), target_col] = y_pred
    return df[target_col]


if __name__ == "__main__":
    data = pd.read_csv(os.path.join(csv_dir, 'personality_dataset.csv'))

    # Define target columns with missing values
    numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
                'Friends_circle_size', 'Post_frequency']

    categorical_cols = ['Stage_fear', 'Drained_after_socializing']

    # Encode categorical columns temporarily for regression
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        notnull_mask = data[col].notnull()
        data.loc[notnull_mask, col] = le.fit_transform(data.loc[notnull_mask, col])
        label_encoders[col] = le

    # Convert object to float for encoded columns
    data[categorical_cols] = data[categorical_cols].astype('float')

    # Final check
    print(data.isnull().sum())
