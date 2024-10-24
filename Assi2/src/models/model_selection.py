import joblib
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR


def train_models(X_train, y_train):
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'AdaBoost': AdaBoostRegressor(),
        'K-Nearest Neighbors': KNeighborsRegressor(),
        'Support Vector Regressor': SVR()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_train)
        mse = mean_squared_error(y_train, predictions)
        results[name] = mse
        print(f'{name} - MSE: {mse}')

    best_model_name = min(results, key=results.get)
    print(f'Best Model: {best_model_name}')

    return models[best_model_name]


def save_model(model, filename):
    joblib.dump(model, filename)
