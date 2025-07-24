import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Encode categorical columns
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])

    symptom_cols = [col for col in df.columns if col.startswith('symptom')]
    for col in symptom_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    df['disease'] = le.fit_transform(df['disease'])

    X = df.drop('disease', axis=1)
    y = df['disease']

    return X, y, le
