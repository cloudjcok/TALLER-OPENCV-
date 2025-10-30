from django.contrib import admin
from django.urls import path
import detector.views as views

urlpatterns = [
    path('', views.index, name='index'), 
    path('video-feed/', views.video_feed, name='video_feed'),
    path('stop-camera/', views.stop_camera, name='stop_camera'),
]

