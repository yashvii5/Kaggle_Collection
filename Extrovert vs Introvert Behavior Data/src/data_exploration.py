# Importing Libraries
import os
import pandas as pd

# Path to Directories
cur_dir = os.path.dirname(os.path.abspath(__file__))
csv_dir = os.path.join(cur_dir, '..', 'data', 'personality_dataset.csv')

if __name__ == "__main__":
    data = pd.read_csv(csv_dir)
    print(f'The Personality Dataset contains {data.shape[1]} columns and {data.shape[0]} rows.')
    print(f'The seven dependent variable names are:\n {data.columns[:-1]}')
    print(f'The target variable names is:\n {data.columns[-1]}')
    print(f'The Data Description looks like this: {data.describe()}')
    print(f'The Dataset info looks like this: {data.info()}')
    print(f'The Summary of missing values in data: {data.isnull().sum()}')
