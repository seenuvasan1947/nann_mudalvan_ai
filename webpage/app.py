import re
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from flask import Flask, app, request, render_template
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
from tensorflow.python.ops.gen_array_ops import concat
from tensorflow.keras.models import load_model
app=Flask(__name__)
model=load_model('crime.h5',compile=False)
app=Flask(__name__)
#home page
@app.route('/')
def home():
    return render_template('home.html')
#prediction page
@app.route('/prediction')
def prediction():
    return render_template('predict.html')
@app.route('/predict', methods=['POST']) 
def predict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files ['image']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename (f.filename))
        f.save(file_path)
        img = image.load_img(file_path, target_size=(64, 64))
        x = image.img_to_array(img) # Converting image into array
        x = np.expand_dims (x, axis=0) # expanding Dimensions 
        pred = np.argmax (model.predict(x)) # Predicting the higher probablity index
        op= ['Fighting','Arrest', 'Vandalism', 'Assault', 'Stealing', 'Arson', 'NormalVideos', 'Burglary', 'Explosion', 'Robbery', 'Abuse', 'Shooting', 'Shoplifting', 'RoadAccidents']
        op [pred]
        result = op[pred]
        result='The predicted output is {}'.format(str(result))
        print (result)
    return render_template('predict.html',text=result)
if __name__=="__main__":
    app.run(debug=True)