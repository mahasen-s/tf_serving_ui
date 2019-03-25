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
from base64 import decodestring, b64encode, b64decode
import os
import flask
import redis
import requests
import pickle
import cv2




app = dash.Dash(__name__)
cache = redis.Redis(host='redis', port = 6379)
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
        #html.Pre(pred, style={
        #    'whiteSpace': 'pre-wrap',
        #    'wordBreak': 'break-all'
        #}),
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src='data:image/png;base64,{}'.format(contents)),
        html.Hr(),
    ], style={'textAlign':'center'})


@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename')])
def update_output(list_of_contents, list_of_names):
    processed_imgs = []
    if list_of_contents is not None:
        for ind, img in enumerate(list_of_contents):
            
            if cache.exists(img) == False:
                # Hacky
                imgstr = img.split(',')[1]
                imgbytes = Image.open(BytesIO(b64decode(imgstr)))
                imgarr = np.asarray(imgbytes)

                # Get predictions
                pred = get_predictions(imgarr)
                imgpred = draw_preds(imgarr, pred)

                # Add to cache
                cache.set(img, pickle.dumps(imgpred))
            
            # Read through cache
            processed_imgs.append(pickle.loads(cache.get(img)))

        # Update html
        children = [
            parse_contents(content, name) for content, name in
            zip(processed_imgs, list_of_names)]
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

def predict_json_payload(img_str):
    # Formats the json payload for the predict TF Serving API
#    image = imread(io.BytesIO(base64.b64decode(img_str)))
    image = Image.open(io.BytesIO(img_str))
    payload = {"instances": [image.tolist()] }
    return payload

def get_predictions(imgarr, min_acc=0.5):
    # Returns predictions from TF Serving as JSON

    # Formulate REST payload
    #image = np.array(Image.open(filename)).tolist()
    payload = {"instances": [imgarr.tolist()] }
    req = requests.post('http://tf_serving:8080/v1/models/default:predict', json=payload)

    # Process return
    raw_preds = req.json()['predictions'][0]
    keep_list =  np.where(np.array(raw_preds['detection_scores'])>min_acc)[0]

    preds = {y: [raw_preds[y][x] for x in keep_list] for y in ['detection_boxes', 'detection_scores', 'detection_classes']}
    preds['num_detections'] =  len(keep_list)

    return preds

def spec(N):                                             
    t = np.linspace(-510, 510, N)                                              
    return np.round(np.clip(np.stack([-t, 510-np.abs(t), t], axis=1), 0, 255)).astype(np.uint8)

def draw_preds(imgarr, preds):
    # Draws predicted boxes on image
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get labels
    # Load catergory labels
    labels_path= './model/labels.txt'
    with open(labels_path, 'r') as f:
        labels = f.read()
    labels = {k+1:v for k,v in enumerate(labels.split())}

    # Load image, this might not do anything for pure numpy arrays
    img = cv2.cvtColor(imgarr[:,:,(2,1,0)], cv2.COLOR_BGR2RGB)
    
    
    h, w = img.shape[:2]
    
    n_cats = len(labels)
    cmap = spec(n_cats)
    
    for i in range(preds['num_detections']):
        det_class = int(preds['detection_classes'][i])
        det_score = preds['detection_scores'][i] 
        det_bbox  = preds['detection_boxes'][i] 
        det_label = labels[det_class]
        
        # draw box
        p1 = (int(det_bbox[1] * w), int(det_bbox[0] * h)) 
        p2 = (int(det_bbox[3] * w), int(det_bbox[2] * h))
        
        col= tuple([int(x) for x in cmap[det_class]])
        cv2.rectangle(img, p1, p2, col, thickness=2)
        
        # label box
        cv2.putText(img,'%s, %4.2f'%(det_label, det_score),
                    p1, font, 1, col, 2, cv2.LINE_AA)
    
    # Change colorspace before recoding
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Process into base64 string
    retval, buffer = cv2.imencode('.jpg', img)
    imgtxt = b64encode(buffer).decode()
    return imgtxt
    
    

if __name__ == '__main__':
    app.run_server(host='0.0.0.0',debug=True)
