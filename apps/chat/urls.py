from django.urls import path
from . import views

urlpatterns = [
    path('', views.rag_agent_home, name='rag_agent_home'),
    path('send_message/', views.send_message, name='send_message'),
]