from django.urls import path
from .views import predict_sales, predict_api, sales_plot

urlpatterns = [
    path('', predict_sales),
    path('api/predict/', predict_api),
    path('plot/', sales_plot),
]
