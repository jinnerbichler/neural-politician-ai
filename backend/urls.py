from django.conf.urls import url

from backend import views

urlpatterns = [
    url(r'^generate-speech', views.generate_speech, name='generate_speech')
]