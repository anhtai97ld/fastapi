import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

def predict_house_prices(df: pd.DataFrame):
    selected_columns = [
        'OverallQual', 'GrLivArea', 'TotalBsmtSF', '2ndFlrSF', 'BsmtFinSF1',
        '1stFlrSF', 'GarageCars', 'GarageArea', 'LotArea', 'YearBuilt',
        'YearRemodAdd', 'Neighborhood', 'TotRmsAbvGrd', 'SalePrice'
    ]

    # Giữ lại các cột cần thiết nếu có
    df = df[[col for col in selected_columns if col in df.columns]]

    # Handle missing values
    df = df.fillna(0)

    # Encode Neighborhood nếu có
    if 'Neighborhood' in df.columns:
        le = LabelEncoder()
        df['Neighborhood'] = le.fit_transform(df['Neighborhood'])

    # Thêm các cột bổ sung
    df['OverHouse_AgeallQual'] = 0
    df['Remod_Age'] = 0
    df['Total_Bathrooms'] = 0
    df['Total_Porch_SF'] = 0
    df['HasGarage'] = 0

    # Xử lý drop cột target nếu có
    if 'SalePrice' in df.columns:
        df = df.drop('SalePrice', axis=1)

    # Load model
    model = joblib.load('./house_price_model.pkl')

    # Predict
    try:
        predictions = model.predict(df)
        return predictions
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None
