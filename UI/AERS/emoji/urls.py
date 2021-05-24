from django.urls import path
from emoji import views
urlpatterns = [
    path('', views.index, name = "index")
]