import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
#import dash_table_experiments as dt                                                                                                                                                           


import datetime
import json
import pandas as pd
import plotly
import io
import numpy as np
from io import BytesIO
from PIL import Image
from base64 import decodestring
import os
import flask

app = dash.Dash(__name__)

app.scripts.config.serve_locally = True

app.layout = html.Div([
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-image-upload'),
])


def parse_contents(contents, filename):
    return html.Div([
        html.H3(filename),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        html.Hr(),
    ], style={'textAlign':'center'})


@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename')])
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [
            parse_contents(content, name) for content, name in
            zip(list_of_contents, list_of_names)]
        return children


css_directory = os.getcwd()
stylesheets = ['assets/default.css']
static_css_route = '/static/'


@app.server.route('{}<stylesheet>'.format(static_css_route))
def serve_stylesheet(stylesheet):
    if stylesheet not in stylesheets:
        raise Exception(
            '"{}" is excluded from the allowed static files'.format(
                stylesheet
            )
        )
    return flask.send_from_directory(css_directory, stylesheet)


for stylesheet in stylesheets:
    app.css.append_css({"external_url": "/static/{}".format(stylesheet)})

def imstr2np(im_str):
    # Converts a base64 encoded string to a numpy array
    image = Image.open(BytesIO(decodestring(im_str)))
    return np.array(image)

def predict_json_payload(img_str_list):
    # Formats the json payload for the predict TF Serving API
    payload = {"instances": [imstr2np(x).tolist() for x in img_str_list]}
    return payload

if __name__ == '__main__':
    app.run_server(debug=True)
