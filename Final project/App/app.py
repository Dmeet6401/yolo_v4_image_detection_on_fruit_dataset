# save this as app.py
from flask import Flask, request, jsonify
import tensorflow as tf
import cv2 as cv
from PIL import Image
import numpy as np


app = Flask(__name__)
model=tf.keras.models.load_model('VGG16_scratch_150_0.88.h5')
classes = ['Buildings', 'Forest', 'Glaciers', 'Mountains', 'Sea', 'Street']

# UPLOAD_FOLDER = r'D:\E\DXAssignment\flask\IntelClassificationAPI\upload'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def hello():
    return '''
    <h1>Intel image Classification API</h1>
    <h2> Available Classes are : </h2>
    <ul>
    <li>Buildings</li>
    <li>Forest</li>
    <li>Glaciers</li>
    <li>Mountains</li>
    <li>Sea</li>
    <li>Street</li>
    </ul>
    <br>
    <br>
    <h2> To classify an image send post request to /predict </h2>
    <ul>
    <li>key : value  ==> inp_img : file with {'png', 'jpg', 'jpeg'} extension </li>
    </ul>
    '''


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'inp_img' not in request.files:
            dictToReturn = {'Wrong key name': "Error : make your key name as inp_img."}
            return jsonify(dictToReturn)

        file = request.files['inp_img']

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            dictToReturn = {'No File is choosen': "Error : make sure to choose an image."}
            return jsonify(dictToReturn)
        if file and allowed_file(file.filename):
            # filename = secure_filename(file.filename)
            # path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # file.save(filename) # file will be saved.
            
            pil_image = Image.open(file)
            img = np.array(pil_image)
            
            # img = cv.imread(img)
            # os.remove(filename)
            IMG_SIZE = (150,150) # for VGG16
            img = cv.resize(img,IMG_SIZE)
            img=img/255.0
            img = np.expand_dims(img, axis=0)
            predict_x= model.predict(img)

            li=[]
            for val in predict_x[0]:
                li.append(float(val)*100)

            dir_clases = dict(zip(li,classes))

            return jsonify(dir_clases)
        else:
            dictToReturn = {'Wrong File Extension': "Error : choose file with {'png', 'jpg', 'jpeg'} extension."}
            return jsonify(dictToReturn)


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080)