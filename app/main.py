import streamlit as st
import os
import imageio
import numpy as np
import tensorflow as tf
from modelutil import load_model # loads model and weights
from utils import load_data, num_to_char # data loading
import tempfile
st.title('LipNet Demo')

video_file = st.file_uploader("Upload a video", type=['mpg'])

if video_file:
    video_path = os.path.join(tempfile.gettempdir(), video_file.name)

    with open(video_path, 'wb') as f:
        f.write(video_file.getbuffer())

        # Load and preprocess video
    video, _ = load_data(video_path)
    video = np.expand_dims(video, axis=0)

    # Load model and predict
    model = load_model()
    preds = model.predict(video)

    # Decode predictions
    decoder = tf.keras.backend.ctc_decode(preds, input_length=[75], greedy=True)[0][0]
    prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')

    # Display video and predicted text
    col1, col2 = st.columns(2)
    with col1:
        st.video(video_path)
    with col2:
        st.text(prediction)

# pip install opencv-python matplotlib imageio gdown tensorflow streamlit flask pyngrok