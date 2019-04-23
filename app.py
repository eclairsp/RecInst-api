from flask import Flask, jsonify, request, send_file, url_for
from flask_restful import Api, Resource, reqparse
import werkzeug
from werkzeug.utils import secure_filename
import os
import uuid
from flask_cors import CORS
from spectrogram import convert_to_spectrogram as cts
from keras.models import Sequential
from keras.models import load_model
import tensorflow as tf
import keras
import json
from shutil import copy2

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

global model
model = load_model("cnn-med-model-backtothefututre.h5")
global graph
graph = tf.get_default_graph()

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)

UPLOAD_FOLDER = os.path.basename('uploads')
STATIC = os.path.basename('static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['wav', 'mp3', 'flac'])

def predict(file):
    import numpy as np
    from keras.preprocessing import image
    test_image = image.load_img(file, target_size = (128, 128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    with graph.as_default():
        result = model.predict_proba(test_image)
    op = {
        'gac' : float("%0.5f" % (result[0][0].item())),
        'gel' : float("%0.5f" % (result[0][1].item())),
        'org' : float("%0.5f" % (result[0][2].item())),
        'pia' : float("%0.5f" % (result[0][3].item())),
        'voi' : float("%0.5f" % (result[0][4].item()))
    } # '.item() helps with result number being float32 and cant be used in json'
    return op

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class Process(Resource):
    def post(self):
        data = request.get_json()
        filename = data['filename']
        spectrogram = os.path.basename('spectrograms')
        fileSrc = os.path.join(UPLOAD_FOLDER, filename)
        path = cts(fileSrc, spectrogram, data['filename'])
        path = path + ".png"
        res = predict(path)
        name = {
            'message' : 'Converted',
            'result' : res
        }
        return jsonify(name)

class Upload(Resource):
    def post(self):
        file = request.files['files']
        if file.filename == '':
            message = {
                'message' : 'false'
            }

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filename = str(uuid.uuid4().hex) + '-' + filename.rsplit('.',1)[0] +  '.' + filename.rsplit('.',1)[1].lower()
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            copy2((os.path.join(app.config['UPLOAD_FOLDER'], filename)), STATIC)
            message = {
                'filename_server' : filename,
                'filename_user' : filename.split('-', 1)[1],
                'message' : 'true',
            }
        else:
            message = {
                'message' : 'extension'
            }
            return message

        return jsonify(message)

api.add_resource(Process, '/process')
api.add_resource(Upload, '/upload')

if __name__ == '__main__':
    app.run()