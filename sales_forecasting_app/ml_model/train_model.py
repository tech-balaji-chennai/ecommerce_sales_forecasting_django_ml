import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

data = pd.read_csv(r"..\data\sales_data_sample.csv")

X = data[['marketing_spend', 'season', 'product_type']]
y = data['sales']

model = LinearRegression()
model.fit(X, y)
joblib.dump(model, "sales_forecasting_model.pkl")
