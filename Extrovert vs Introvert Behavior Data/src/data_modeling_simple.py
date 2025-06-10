import os
import pandas as pd
from sklearn.impute import SimpleImputer

# Path to Directories
cur_dir = os.path.dirname(os.path.abspath(__file__))
csv_dir = os.path.join(cur_dir, '..', 'data')


if __name__ == "__main__":
    data = pd.read_csv(os.path.join(csv_dir, 'personality_dataset.csv'))  # replace with actual file path

    # Separate numerical and categorical columns
    numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
                    'Friends_circle_size', 'Post_frequency']
    categorical_cols = ['Stage_fear', 'Drained_after_socializing']

    # Median imputer for numerical columns
    numeric_imputer = SimpleImputer(strategy='median')
    data[numeric_cols] = numeric_imputer.fit_transform(data[numeric_cols])

    # Mode imputer for categorical columns
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])

    # Check to make sure missing values are handled
    print(data.isnull().sum())

    # Save data as csv file
    data.to_csv(os.path.join(csv_dir, 'simpleImputer.csv'))


