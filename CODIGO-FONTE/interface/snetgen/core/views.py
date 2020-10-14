from django.views.generic import FormView, TemplateView
from django.urls import reverse_lazy

import csv

from .forms import ConfigForm


class ConfigFormView(FormView):
    form_class = ConfigForm
    success_url = reverse_lazy('success')
    template_name = 'crispy_form.html'

    def form_valid(self, form):

        data = form.cleaned_data

        hash_id = abs(hash(tuple(data[k] for k in sorted(data.keys()))))

        with open(str( hash_id ), 'w', newline='') as data_buffer:
            file_writer = csv.writer(data_buffer)
            file_writer.writerow(data[k] for k in sorted(data.keys()))
            file_writer.writerow(k for k in sorted(data.keys()))

        return super().form_valid(form)


class SuccessView(TemplateView):
    template_name = 'success.html'
