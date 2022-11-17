import json
import requests
import numpy as np
import os
from PIL import Image
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename, redirect

from keras.models import load_model
from keras.preprocessing import image
from flask import send_from_directory

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "#My_APIKEY"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}



UPLOAD_FOLDER = 'C:\\Users\\Samprokshana\\Desktop\\Project Development\\Sprint-3\\uploads'


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model(".\static\mnistCNN.h5")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == "POST":
        f = request.files["image"]
        filepath = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filepath))

        upload_img = os.path.join(UPLOAD_FOLDER, filepath)
        img = Image.open(upload_img).convert("L")  # convert image to monochrome
        img = img.resize((28, 28))  # resizing of input image

        im2arr = np.array(img)  # converting to image
        im2arr = im2arr.reshape(1, 28, 28, 1)  # reshaping according to our requirement
        

        # NOTE: manually define and pass the array(s) of values to be scored in the next line
        response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/c7e81304-ffeb-481c-b731-7527336e8e94/predictions?version=2022-11-15',
         headers={'Authorization': 'Bearer ' + mltoken})
        print("Scoring response")
        print(response_scoring.json())


        pred = model.predict(im2arr)

        num = np.argmax(pred, axis=1)  # printing our Labels
    

    return render_template('predict.html', num=str(num[0]))


if __name__ == '__main__':
    app.run(debug=True, threaded=False)