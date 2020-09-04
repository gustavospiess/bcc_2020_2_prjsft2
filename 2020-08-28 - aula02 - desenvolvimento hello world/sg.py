import PySimpleGUI as sg
sg.theme('SystemDefault')

layout = [  [sg.Text('Text:'), sg.InputText()],
            [sg.Text('File name: '), sg.Text('', size=(15,1)), sg.FilesBrowse()],
            # [sg.Text('Date: '), sg.In('', size=(20,1)), sg.CalendarButton('Choose Date', target=(2,1), key='date')],
            [sg.Text('Slider: '), sg.Slider(range=(1,12), key='_SLIDER_', orientation='h')],
            [sg.Multiline(size=(70,4),key='AUSGABE', default_text='try typing here!\nMutiline text, press ^R to reformat.\n')],
            [sg.Text('Pick one: ')],
            [sg.Radio('Option1', 'RADIO1')],
            [sg.Radio('Option2', 'RADIO1')],
            [sg.Radio('Option3', 'RADIO1')],
            [sg.Text('Pick several: ')],
            [sg.Checkbox('Option1', key = 'CB1')],
            [sg.Checkbox('Option2', key = 'CB2')],
            [sg.Checkbox('Option3', key = 'CB3')],
            [sg.Ok(key='OK')]]

window = sg.Window('Welcome to Npyscreen', layout)

while True:             
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'OK'):
        break
    elif event:
        print(event)

window.close()
