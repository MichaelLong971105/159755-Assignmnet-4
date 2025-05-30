import os
import pandas as pd
import joblib
import numpy as np
import sqlite3
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def train_model():
    base_path = os.path.dirname(os.path.abspath(__file__))
    database_path = os.path.join(base_path, '..', 'data', 'allData.sl3')
    conn = sqlite3.connect(database_path)
    df = pd.read_sql_query("SELECT * FROM player_stats WHERE value != 0 and wage != 0", conn)
    conn.close()

    df['value'] = np.log(df['value'])
    df['wage'] = np.log(df['wage'])

    X = df.drop(columns=['full_name', 'value', 'wage', 'year'])
    y = df['value']

    if 'best_position' in X.columns:
        X = pd.get_dummies(X, columns=['best_position'])

    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype(str).str.extract(r'(\d+)')
            X[col] = pd.to_numeric(X[col], errors='coerce')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # y_perd = model.predict(X_test)
    # mse = mean_squared_error(y_test, y_perd)
    # print(mse)
    joblib.dump(model, os.path.join(base_path, '..', 'models', 'player_value_model.pkl'))


def predict_player_value(player_dict):
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, '..', 'models', 'player_value_model.pkl')
    try:
        model = joblib.load(model_path)
    except (FileNotFoundError, EOFError, Exception):
        print("Training model...")
        train_model()
        model = joblib.load(model_path)

    df = pd.DataFrame([player_dict])
    for col in ['Stamina', 'Dribbling', 'Short passing']:
        df[col] = df[col].astype(float)

    if 'Best position' in df.columns:
        df = pd.get_dummies(df)

    model_columns = model.feature_names_in_
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[model_columns]

    log_pred = model.predict(df)[0]
    return np.exp(log_pred)


if __name__ == '__main__':
    train_model()

