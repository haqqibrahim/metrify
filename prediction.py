from sklearn.preprocessing import StandardScaler

def make_predictions(model, X):
    # Standardize the data
    sc = StandardScaler()
    X_standardized = sc.fit_transform(X)

    # Make predictions
    predictions = model.predict(X_standardized)

    return predictions
