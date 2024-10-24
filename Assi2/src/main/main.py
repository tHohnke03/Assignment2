# main.py

import pandas as pd
from src.ingestion.data_preprocessing import load_data, preprocess_data
from src.gui.plot_visualisation import plot_correlation_matrix
from src.models.model_selection import train_models
from src.utilities.utils import get_selected_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    file_path = '/Users/trenthohnke/Downloads/Assi2/resources/Australian_Vehicle_Prices.csv'
    data = load_data(file_path)
    data = preprocess_data(data)

    # Visualize data
    plot_correlation_matrix(data)

    target_variable = 'Price'
    selected_features = get_selected_features(data, target_variable)

    X = data[selected_features]
    y = data[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize data (optional)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train models
    train_models(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
