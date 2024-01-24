import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Encoding ordinal columns
    enc = LabelEncoder()
    for i in (2, 3, 4, 5, 6, 7, 16, 26):
        data.iloc[:, i] = enc.fit_transform(data.iloc[:, i])

    # Drop unnecessary columns
    data.drop(['EmpNumber'], inplace=True, axis=1)

    # Select important columns
    X = data.iloc[:, [4, 5, 9, 16, 20, 21, 22, 23, 24]]

    return X
