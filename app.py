import av
import cv2
import numpy as np
from keras.models import model_from_json
import streamlit as st
from streamlit_webrtc import webrtc_streamer

mask_dict = {0: "Masked", 1: "Unmasked"}
emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Surprised"}

@st.cache_resource
def load_maskDetectionModel():
    json = open('models/MaskModel.json', 'r')
    model = json.read()
    json.close()
    loadedModel = model_from_json(model)
    loadedModel.load_weights('models/MaskModel.h5')
    return loadedModel

@st.cache_resource
def load_unmaskedEmotionModel():
    json = open('models/EmotionUnmaskedModel.json', 'r')
    model = json.read()
    json.close()
    loadedModel = model_from_json(model)
    loadedModel.load_weights('models/EmotionUnmaskedModel.h5')
    return loadedModel

@st.cache_resource
def load_maskedEmotionModel():
    json = open('models/EmotionMaskedModel.json', 'r')
    model = json.read()
    json.close()
    loadedModel = model_from_json(model)
    loadedModel.load_weights('models/EmotionMaskedModel.h5')
    return loadedModel

@st.cache_resource
def load_faceDetectionModel():
    face_cascade = cv2.CascadeClassifier('models/HaarCascadeFiles/haarcascade_frontalface_default.xml')
    return face_cascade

maskModel = load_maskDetectionModel()
unmaskedEmotion = load_unmaskedEmotionModel()
maskedEmotion = load_maskedEmotionModel()
faceCascade = load_faceDetectionModel()

ds_factor = 0.6
icon_size = 20

print("Models have been loaded.")

st.title("Facial Emotion Recognition App")
st.write("This web app is a demonstration of an ML model trained to detect emotions of masked/unmasked faces.")
st.write("It can tell between two face states: masked/unmasked, and four emotions: neutral/happy/angry/surprised.")
st.write("To begin, click START below. (It may take up to 1-2 minutes to start the stream depending on your location).")

def callback(frame):
    img = frame.to_ndarray(format="bgr24")

    img = cv2.resize(img, (720, 405))
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    num_faces = faceCascade.detectMultiScale(gray_frame, 1.3, 5)

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
            UnmaskedEmotionPredict = unmaskedEmotion.predict(cropped_img_gray)
            maxindex2 = int(np.argmax(UnmaskedEmotionPredict))
            cv2.putText(img, emotion_dict[maxindex2], (x + 200, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, mask_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            continue

        elif mask_dict[maxindex]=='Masked':
            cv2.rectangle(img, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
            MaskedEmotionPredict = maskedEmotion.predict(cropped_img_gray)
            maxindex3 = int(np.argmax(MaskedEmotionPredict))
            cv2.putText(img, emotion_dict[maxindex3], (x + 200, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, mask_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            continue
        
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="face-emotion-recognition",
    rtc_configuration={ "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=callback,
    media_stream_constraints={"video": True, "audio": False},
    )

st.write("Github: https://github.com/ITM2020/masked-emotion-recognition-app")