# import model # Import the python file containing the ML model
from flask import Flask, request, render_template, jsonify  # Import flask libraries
import pickle as pk
import numpy as np
import os
import json
import requests
import pandas as pd
import threading
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import classification_report,accuracy_score
acc = 0.8
def updatemodel():
    while True:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        if current_time == '01:00:00':
            df = pd.read_json('data.json')
            X = df['imgData']
            X = pd.DataFrame(X.tolist(), index= X.index)
            y = df['class']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
            model = LogisticRegression().fit(X_train,y_train)
            # model.predict()
            y_pre = model.predict(X_test)
            if accuracy_score(y_pre, y_test) > acc:
                acc = accuracy_score(y_pre, y_test)
                filename = 'model.sav'
                pickle.dump(model, open(filename, 'wb'))
# Initialize the flask class and specify the templates directory
app = Flask(__name__, template_folder="templates")

# Load the model
model = pk.load(open('model.sav', 'rb'))

# Dictionary containing the mapping
# variety_mappings = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

# Function for classification based on inputs
# def classify(a, b, c, d):
#     print('\n classifier is called \n')
#     arr = np.array([a, b, c, d]) # Convert to numpy array
#     arr = arr.astype(np.float64) # Change the data type to float
#     query = arr.reshape(1, -1) # Reshape the array
#     prediction = variety_mappings[loaded_model.predict(query)[0]] # Retrieve from dictionary
#     return prediction # Return the prediction

# Default route set as 'home'


@app.route('/')
def home():
    return render_template('home.html')  # Render home.html

# Route 'classify' accepts GET request


@app.route('/classify', methods=['POST','GET'])
def classify():
    # try:
        if request.method == 'POST':
        # df = pd.read_json(request.json)
        # X = df['imgData']
        # X = pd.DataFrame(X.tolist(), index=X.index)
        # print(type(request.json['imgdata']))
            img = request.json['imgdata']
            print(np.max(model.predict_proba([img])))
            # ,"proba":np.max(model.predict_proba([img])))
            return jsonify({"resalt":model.predict([img])[0],"proba":np.max(model.predict_proba([img]))})
    # except:
    #     return 'Error'
    

# Run the Flask server
if(__name__=='__main__'):
    threading.Thread(target=updatemodel).start()
    app.run(debug=True)        