from django.http import HttpResponseRedirect
from django.urls import path
from django.views.generic import RedirectView

from snetgen.core import views

urlpatterns = [
    path('', views.ConfigFormView.as_view(), name='config_form'),
    path('config/', views.ConfigFormView.as_view(), name='config_form'),
    path('success/', views.SuccessView.as_view(), name='success'),
]
