from django.urls import path
from . import views

urlpatterns = [
    # Página principal
    path('', views.index, name='index'),
    
    # Streaming de la cámara
    path('video_feed_cam/', views.video_feed_cam, name='video_feed_cam'),
]
