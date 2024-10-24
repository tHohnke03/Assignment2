import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


def preprocess_data(data):
    """Preprocess the dataset."""
    # Remove duplicates
    data = data.drop_duplicates()

    # Clean up column names
    data.columns = data.columns.str.strip()

    # Replace placeholders with NaN
    data.replace({'-': np.nan, '- / -': np.nan}, inplace=True)

    # Convert 'Price' to numeric, coercing errors to NaN
    data['Price'] = pd.to_numeric(data['Price'], errors='coerce')

    # Drop rows with NaN values in 'Price'
    data = data.dropna(subset=['Price'])

    # Handle specific columns
    columns_to_extract = {
        'FuelConsumption': r'(\d+\.\d+)',
        'CylindersinEngine': r'(\d+)'
    }

    for column, regex in columns_to_extract.items():
        if column in data.columns:
            data[column] = data[column].str.extract(regex)[0].astype(float)
        else:
            print(f"Error: '{column}' column not found.")

    # Check for other columns
    optional_columns = ['Kilometres', 'CylindersinEngine', 'Doors', 'Seats']
    for column in optional_columns:
        if column not in data.columns:
            print(f"Warning: '{column}' not found in DataFrame.")

    # Convert categorical variables to dummy variables
    data = pd.get_dummies(data, drop_first=True)

    return data


def plot_target_distribution(data):
    plt.hist(data['Price'], bins=30, edgecolor='k')
    plt.title('Distribution of Price')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.show()
