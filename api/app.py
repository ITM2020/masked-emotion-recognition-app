import cv2
import numpy as np
from keras.models import model_from_json

import streamlit as st
from streamlit_webrtc import webrtc_streamer

import av

mask_dict = {0: "Masked", 1: "Unmasked"}
emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Surprised"}

# Load masked/unmasked model
json_file = open('api/models/MaskModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
maskModel = model_from_json(loaded_model_json)
maskModel.load_weights('api/models/MaskModel.h5')

# Load unmasked emotion recognition model
json_file = open('api/models/EmotionUnmaskedModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
UnmaskedEmotion = model_from_json(loaded_model_json)
UnmaskedEmotion.load_weights('api/models/EmotionUnmaskedModel.h5')

# Load masked emotion recognition model
json_file = open('api/models/EmotionMaskedModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
MaskedEmotion = model_from_json(loaded_model_json)
MaskedEmotion.load_weights('api/models/EmotionMaskedModel.h5')

# Load HaarCascade face classifier
face_cascade = cv2.CascadeClassifier('api/models/HaarCascadeFiles/haarcascade_frontalface_default.xml')
ds_factor = 0.6

print("Models have been loaded.")

st.title("Facial Emotion Recognition of (Un)Masked Faces")
st.write("To begin detecting the whether a person is wearing a mask or not and their facial expression, click the START button below.")

def callback(frame):
    img = frame.to_ndarray(format="bgr24")

    img = cv2.resize(img, (1280, 720))
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    num_faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    for (x, y, w, h) in num_faces:
        roi_frame = img[y:y + h, x:x + w]
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_frame, (128, 128)), -1), 0)
        cropped_img_gray = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict whether wearing a mask
        maskPredict = maskModel.predict(cropped_img)
        maxindex = int(np.argmax(maskPredict))

        # use mask or unmasked set to recognize emotion depending on masked/unmasked
        if mask_dict[maxindex]=='Unmasked':
            cv2.rectangle(img, (x, y - 50), (x + w, y + h + 10), (0, 0, 255), 4)
            UnmaskedEmotionPredict = UnmaskedEmotion.predict(cropped_img_gray)
            maxindex2 = int(np.argmax(UnmaskedEmotionPredict))
            cv2.putText(img, emotion_dict[maxindex2], (x + 200, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, mask_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            continue

        elif mask_dict[maxindex]=='Masked':
            cv2.rectangle(img, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
            MaskedEmotionPredict = MaskedEmotion.predict(cropped_img_gray)
            maxindex3 = int(np.argmax(MaskedEmotionPredict))
            cv2.putText(img, emotion_dict[maxindex3], (x + 200, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, mask_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            continue
        
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="example", video_frame_callback=callback)