#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 22:34:47 2022

@author: j.v.thomasabraham
"""

import requests
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

import numpy as np
import pandas as pd
import os
import tensorflow as tf

from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session

app = Flask(__name__)

fruitmodel = load_model("fruit.h5")
vegmodel = load_model("vegetable.h5")

#render home page
@app.route('/')
def home():
    return render_template('home.html')

#render predict page
@app.route('/prediction')
def prediction():
    return render_template('predict.html')


#prediction and recommendation
@app.route('/predict',methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(filepath)
        testImage = image.load_img(filepath, target_size=(128,128))
        
        x = image.img_to_array(testImage)
        x = np.expand_dims(x, axis=0)
        plant=request.form['plant']
        print(plant)
        
        if(plant=="fruit"):
            preds = fruitmodel.predict(x)
            preds = np.argmax(preds,axis=1)
            df = pd.read_excel('precautions - fruits.xlsx')
            text=str(df.iloc[preds])
            print(df.iloc[preds]) 
                  
        else:
            preds = vegmodel.predict(x)
            preds = np.argmax(preds, axis = 1)
            df = pd.read_excel('precautions - veg.xlsx')
            text=str(df.iloc[preds])
            print(df.iloc[preds]) #['caution'])


    return text
        
if __name__ == "__main__":
    app.run(debug=True, port=8000)