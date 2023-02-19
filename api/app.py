import cv2
import numpy as np
from keras.models import model_from_json

import streamlit as st
from streamlit_webrtc import webrtc_streamer

import av

mask_dict = {0: "Masked", 1: "Unmasked"}
emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Surprised"}

# Load masked/unmasked model
json_file = open('./models/MaskModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
maskModel = model_from_json(loaded_model_json)
maskModel.load_weights("./models/MaskModel.h5")

# Load unmasked emotion recognition model
json_file = open('./models/EmotionUnmaskedModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
UnmaskedEmotion = model_from_json(loaded_model_json)
UnmaskedEmotion.load_weights("./models/EmotionUnmaskedModel.h5")

# Load masked emotion recognition model
json_file = open('./models/EmotionMaskedModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
MaskedEmotion = model_from_json(loaded_model_json)
MaskedEmotion.load_weights("./models/EmotionMaskedModel.h5")

# Load HaarCascade face classifier
face_cascade = cv2.CascadeClassifier("./models/HaarCascadeFiles/haarcascade_frontalface_default.xml")
ds_factor = 0.6

print("Models have been loaded.")

webrtc_streamer(key="example")