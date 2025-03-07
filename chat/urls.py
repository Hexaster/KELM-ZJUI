from django.urls import path
from . import views

app_name = 'chat'
urlpatterns = [
    path('', views.chat_view, name='chat'),
    #path('generate/', views.generate_response, name='generate_response'),
]