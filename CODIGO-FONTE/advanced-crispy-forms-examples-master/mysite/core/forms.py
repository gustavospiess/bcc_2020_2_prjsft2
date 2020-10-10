from django import forms

from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Div, Submit, Row, Column, Field, Fieldset

class ConfigForm(forms.Form):
    n = forms.IntegerField(min_value=20, max_value=250, help_text='Number of vertices')
    max_wth = forms.IntegerField(min_value=1, help_text='Maximum number of edges within a community by vertex')
    max_btw = forms.IntegerField(min_value=1, help_text='Maximum number of edges between a community by vertex')
    mte = forms.IntegerField(help_text='Minimum number of edges in resulting graph')
    k = forms.IntegerField(min_value=2, help_text='Number of communities')
    teta = forms.FloatField(min_value=0, max_value=1, help_text='Threshold for community attributes homogeneity')
    nbRep = forms.IntegerField(min_value=1, help_text='Maximum number of community representatives')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.layout = Layout(
            Fieldset('Dados Gerais',
                'n',
                'max_wth',
                'max_btw',
                'mte',
                'k',
                'teta',
                'nbRep',
                Submit('submit', 'save')
            )
        )
