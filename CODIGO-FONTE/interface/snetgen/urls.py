from django.http import HttpResponseRedirect
from django.urls import path
from django.views.generic import RedirectView


from django.views.generic import FormView, TemplateView

from snetgen.core import views, forms

urlpatterns = [
    path('', views.ConfigFormView.as_view(), name='config_form'),
    path('config/', views.ConfigFormView.as_view(), name='config_form'),
    path('config/<str:hash_id>/1', views.attribute_view_builder, name='att_form_1'),
    path('config/<str:hash_id>/2', views.attribute_view_builder, name='att_form_1'),
    path('success/', views.SuccessView.as_view(), name='success'),
]
