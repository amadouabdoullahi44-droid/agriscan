from django.urls import path
from . import views

urlpatterns = [
    path('', views.analyser_sol, name='analyser_sol'),
]