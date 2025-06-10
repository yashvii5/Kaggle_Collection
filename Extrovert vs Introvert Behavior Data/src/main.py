import os
import pandas as pd

# Path to Directories
cur_dir = os.path.dirname(os.path.abspath(__file__))
csv_dir = os.path.join(cur_dir, '..', 'data')

if __name__ == "__main__":
    simple_data = pd.read_csv(os.path.join(csv_dir, 'simpleImputer.csv'))
    regression_data = pd.read_csv(os.path.join(csv_dir, 'regression.csv'))

    print(simple_data)
    print(regression_data)