from django.shortcuts import render
from django.conf import settings
import numpy as np
import joblib, os

# Loading ml model
model = joblib.load(os.path.join(settings.BASE_DIR, "ml_model/sales_forecasting_model.pkl"))

# For HTML form handling (web users)
def predict_sales(request):
    prediction = None
    if request.method == "POST":
        ms = float(request.POST['marketing_spend'])
        season = int(request.POST['season'])
        pt = int(request.POST['product_type'])
        prediction = round(model.predict([[ms, season, pt]])[0], 2)
    return render(request, "index.html", {"prediction": prediction})

# For API calls (Postman, JavaScript, mobile)
from rest_framework.decorators import api_view
from rest_framework.response import Response

@api_view(['POST'])
def predict_api(request):
    data = request.data
    ms = float(data['marketing_spend'])
    season = int(data['season'])
    pt = int(data['product_type'])
    pred = model.predict([[ms, season, pt]])[0]
    return Response({'predicted_sales': round(pred, 2)})

# Create Plot View
from django.http import HttpResponse
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def sales_plot(request):
    df = pd.read_csv(os.path.join(settings.BASE_DIR, 'data', 'sales_data_sample.csv'))
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.barplot(x="marketing_spend", y="sales", hue="season", data=df)
    plt.title("Sales vs Marketing Spend by Season")
    plot_path = os.path.join(settings.BASE_DIR, 'forecast', 'static', 'sales_forecast_plot.png')
    plot_exists = os.path.exists(plot_path)
    print("Plot exists:", plot_exists)
    plt.savefig(plot_path)
    plt.close()
    return render(request, 'sales_forecast_plot.html', {'plot_exists': plot_exists})
