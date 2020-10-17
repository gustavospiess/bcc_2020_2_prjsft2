from django import forms

from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Div, Submit, Row, Column, Field, Fieldset

class AttributeWidget(forms.MultiWidget):
    def __init__(self, *args, **kwargs):
        widgets = (
                forms.TextInput(),
                forms.NumberInput)
        super().__init__(*args, widgets, **kwargs)

    def decompress(self, value):
        if value:
            return value
        return [None, None]

class AttributeField(forms.MultiValueField):
    def __init__(self, *args, **kwargs):
        fields = (
                forms.CharField(max_length=255, help_text='Name for the attribute'),
                forms.IntegerField(min_value=2, max_value=4, help_text='How many options will be'))
        super().__init__(*args, fields, **kwargs)

    def compress(self, data_list):
        return data_list


def attribute_form_builder(qt_att):
    class AttributeForm(forms.Form):
        att_1 = AttributeField(widget=AttributeWidget, required=True)
        att_2 = AttributeField(widget=AttributeWidget, required=True)
        att_3 = AttributeField(widget=AttributeWidget, required=True)
        att_4 = AttributeField(widget=AttributeWidget, required=True)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.helper = FormHelper()
            self.helper.layout = Layout(
                Fieldset('Atributos e quantidades',
                    *[
                        'att_1',
                        'att_2',
                        'att_3',
                        'att_4',
                        ][:qt_att],
                    Submit('submit', 'save')
                )
            )
    return AttributeForm


class ConfigForm(forms.Form):
    n = forms.IntegerField(min_value=20, max_value=250, help_text='Number of vertices')
    max_wth = forms.IntegerField(min_value=1, help_text='Maximum number of edges within a community by vertex')
    max_btw = forms.IntegerField(min_value=1, help_text='Maximum number of edges between a community by vertex')
    mte = forms.IntegerField(help_text='Minimum number of edges in resulting graph')
    k = forms.IntegerField(min_value=2, help_text='Number of communities')
    teta = forms.FloatField(min_value=0, max_value=1, help_text='Threshold for community attributes homogeneity')
    nb_rep = forms.IntegerField(min_value=1, help_text='Maximum number of community representatives')
    qt_att = forms.IntegerField(min_value=1, max_value=4, help_text='Number of given attributes')

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
                'nb_rep',
                'qt_att',
                Submit('submit', 'save')
            )
        )
