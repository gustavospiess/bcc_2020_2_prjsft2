from django.views.generic import FormView, TemplateView
from django.urls import reverse_lazy

from .forms import ConfigForm


class ConfigFormView(FormView):
    form_class = ConfigForm
    success_url = reverse_lazy('success')
    template_name = 'crispy_form.html'


class SuccessView(TemplateView):
    template_name = 'success.html'
