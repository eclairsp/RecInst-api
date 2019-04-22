from flask import Flask, jsonify, request, send_file
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

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

global model
model = load_model("cnn-med-model-backtothefututre.h5")
global graph
graph = tf.get_default_graph()

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)

UPLOAD_FOLDER = os.path.basename('uploads')
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
        'gac' : result[0][0].item(),
        'gel' : result[0][1].item(),
        'org' : result[0][2].item(),
        'pia' : result[0][3].item(),
        'voi' : result[0][4].item()
    } # '.item() helps with result number being float32 and cant be used in json'
    return op

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class Audio(Resource):
    def post(self):
        data = request.get_json()
        filename = data['filename']
        path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            return send_file(path, attachment_filename=filename)
        except:
            message = {
                'message' : 'not found'
            }
            return message

class Process(Resource):
    def post(self):
        data = request.get_json()
        filename = data['filename']
        spectrogram = os.path.basename('spectrograms')
        fileSrc = os.path.join(UPLOAD_FOLDER, filename)
        print(fileSrc)
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

        print('file', file)

        if file.filename == '':
            message = {
                'message' : 'false'
            }

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filename = str(uuid.uuid4().hex) + '-' + filename.rsplit('.',1)[0] +  '.' + filename.rsplit('.',1)[1].lower()
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
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
api.add_resource(Audio, '/audio')

if __name__ == '__main__':
    app.run()