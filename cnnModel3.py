from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from EfModel import EfficientNet,SqueezeExcitation,MBConv,Swish,Flatten
import DoTest
import trainModel
import pandas as pd
# Keras
'''
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from efficientnet.tfkeras import EfficientNetB7
'''
from PIL import Image
import torch
import torch.nn as nn
from math import ceil
import torchvision.transforms as transforms
#from torch.utils.data import  DataLoader

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
#MODEL_PATH = 'models/effecientnetb0.h5'

# Load your trained model
#model = load_model(MODEL_PATH)
directory_contents = os.listdir('models')
model = torch.load('models/'+str(directory_contents[0]),map_location=torch.device('cpu'))
model.eval()
#model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('models/model_resnet.h5')
#print('Model loaded. Check http://127.0.0.1:5000/')


import numpy as np
'''from keras.preprocessing import image
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart'''




def model_predict_e(img_path,model):
    transform = transforms.ToTensor()
    #test_image = image.load_img(img_path, target_size = (224, 224))
    '''test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)'''
    image = Image.open(img_path)
    image = image.resize((224, 224))
    image = transform(image)
    x = image.unsqueeze(0)
    out = model(x)
    n = float(out.data[0][0])
    if n > 0.5:
        return 'Melanoma'
    else:
        return 'Non-melanoma'







@app.route('/', methods=['GET'])
def myindex():
    # Main page
    return render_template('myindex.html')


#Linking
@app.route('/indexhtml123', methods=['GET', 'POST'])
def index():
      #if request.method == 'POST':
        # do stuff when the form is submitted
        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        # return redirect(url_for('myindex'))
        # show the form, it wasn't submitted
      print("Hello123Function")
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
        preds = model_predict_e(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        return str(preds)
    return Nones

@app.route('/predict1', methods=['GET', 'POST'])
def testresult():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        print(f)
        data = pd.read_csv(f)
        data = data.iloc[1:, 0:2]
        print(data)
        p = request.args.get('tpath')
        print(p)

        # Save the file to ./uploads
        #basepath = os.path.dirname(__file__)
        #print(basepath)
        #file_path = os.path.join(
           # basepath, 'uploads', secure_filename(f.filename))
        #f.save(file_path)
        #print(f)
        directory_contents = os.listdir('.')
        print(directory_contents)
        if 'test_data' in directory_contents:
            acc = DoTest.do_test_model('test_data', data)
            return str(acc)
        else:
            return "test_data folder is not available"


        # Make prediction
        #preds = model_predict_e(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string

    return Nones



@app.route('/predict2', methods=['GET', 'POST'])
def train_model():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        print(f)
        data = pd.read_csv(f)
        data = data.iloc[1:, 0:2]
        print(data)
        p = request.args.get('tpath')
        print(p)
        f=data.to_csv()
        # Save the file to ./uploads
        #basepath = os.path.dirname(__file__)
        #print(basepath)
        #file_path = os.path.join(
           # basepath, 'uploads', secure_filename(f.filename))
        #f.save(file_path)
        #print(f)
        #acc = DoTest.do_test_model('benign', data)
        trainModel.train_model(f)

        # Make prediction
        #preds = model_predict_e(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        return "Training finished "
    return Nones


#@app.route('/go_to_test', methods=['GET'])
#def index1():
    # Main page
 #   return render_template('index1.html')

#@app.route('/go_to_train', methods=['GET'])
#def index2():
    # Main page
 #   return render_template('index2.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
