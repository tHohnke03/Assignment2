import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def encode_categorical_features(data):
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    data = pd.get_dummies(data, columns=categorical_features, drop_first=True)
    return data


def split_data(data, target_variable):
    X = data.drop(columns=[target_variable])
    y = data[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
