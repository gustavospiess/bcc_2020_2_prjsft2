from django.views.generic import FormView, TemplateView
from django.urls import reverse_lazy, reverse

from django.http import HttpResponseRedirect
import django.urls as urls

import json

from .forms import ConfigForm, attribute_form_builder

import os
import datetime
import hashlib


class ConfigFormView(FormView):
    hash_id = None
    form_class = ConfigForm
    template_name = 'crispy_form.html'

    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    def get_success_url(self):
       return f'/config/{self.hash_id}/1'

    def form_valid(self, form):
        data = form.cleaned_data
        data['creation_date'] = datetime.datetime.now().strftime("%m%d%Y%H%M%S")

        sha = hashlib.sha256()
        for key in sorted(data.keys()):
            sha.update(str(key).encode('utf-8'))
            sha.update(str(data[key]).encode('utf-8'))

        self.hash_id = sha.hexdigest()
        data['hash_id'] = self.hash_id

        file_path = f'data/{str(self.hash_id)}'
        if not os.path.isfile(file_path):
            with open(f'data/{str(self.hash_id)}', 'w', newline='') as data_buffer:
                json.dump(data, data_buffer)

        return super().form_valid(form)

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        return kwargs


def attribute_view_builder(*args, hash_id=None, **kwargs):
    print('buiild view')
    __import__('pprint').pprint(args)
    __import__('pprint').pprint(kwargs)
    data = None
    with open(f'data/{str(hash_id)}', 'r', newline='') as data_buffer:
        data = json.load(data_buffer)

    class Dynamic(FormView):
        form_class = attribute_form_builder(qt_att=int(data['qt_att']))
        template_name = 'crispy_form.html'

        def get_success_url(self):
            print('get_success_url ___')
            return f'/config/{hash_id}/2'

        def form_valid(self, form):
            print('form valid ___')
            return super().form_valid(form)

    return Dynamic.as_view()(*args, **kwargs)


class SuccessView(TemplateView):
    template_name = 'success.html'

