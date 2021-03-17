from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
#import numpy as np
import tensorflow as tf

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import Model , load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__,template_folder='templates')


#loading the model
Model = load_model('MMmodel.h5')         # Necessary
#print('Model loaded. Starting service...')

#defining the classes to predict
classes_list_target =(['Action', 'Adventure', 'Animation', 'Biography',
       'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy',
       'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News',
       'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War',
       'Western'])


#print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, Model):
    img = image.load_img(img_path, target_size=(150, 150, 3))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = x/255

    probs = Model.predict(x.reshape(1,150,150,3))
    return probs


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
       
        
         # Make prediction
        #probs = model_predict(file_path , model)
        #pred_class = probs.argmax(axis=-1)            # Simple argmax
        #pr = classes[pred_class[0]]
        #result =str(pr) 
       
        probs = model_predict(file_path , Model)
        pred_class = probs.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   
        pr = classes_list_target[pred_class[0]]
        result =str(pr) 
    
        return result 
    return None


if __name__ == '__main__':
    app.run(debug=True)
