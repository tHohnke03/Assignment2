# utils.py

def get_selected_features(data, target_variable):
    selected_features = [col for col in data.columns if col != target_variable]
    return selected_features
