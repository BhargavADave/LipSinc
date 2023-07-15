from flask import Flask, render_template, request
import cv2
import os
import numpy as np
import tensorflow as tf
from modelutil import load_model # loads model and weights
from utils import load_data, num_to_char # data loading
import tempfile

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    video = request.files['video']

    video_path = os.path.join(tempfile.gettempdir(), video.filename)
    video.save(video_path)

    # Load and preprocess video
    video, _ = load_data(video_path)
    video = np.expand_dims(video, axis=0)

    # Load model and predict
    model = load_model()
    preds = model.predict(video)

    # Decode predictions
    decoder = tf.keras.backend.ctc_decode(preds, input_length=[75], greedy=True)[0][0]
    prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')

    return render_template('predict.html', prediction=prediction)

if __name__ == '__main__':
    app.run()