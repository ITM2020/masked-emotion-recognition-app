import cv2
import numpy as np
from keras.models import model_from_json

mask_dict = {0: "Masked", 1: "Unmasked"}
emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Surprised"}

# Load masked/unmasked model
json_file = open('models/MaskModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
maskModel = model_from_json(loaded_model_json)
maskModel.load_weights("models/MaskModel.h5")

# Load unmasked emotion recognition model
json_file = open('models/EmotionUnmaskedModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
UnmaskedEmotion = model_from_json(loaded_model_json)
UnmaskedEmotion.load_weights("models/EmotionUnmaskedModel.h5")

# Load masked emotion recognition model
json_file = open('models/EmotionMaskedModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
MaskedEmotion = model_from_json(loaded_model_json)
MaskedEmotion.load_weights("models/EmotionMaskedModel.h5")

# Load HaarCascade face classifier
face_cascade = cv2.CascadeClassifier("models/HaarCascadeFiles/haarcascade_frontalface_default.xml")
ds_factor = 0.6

print("Models have been loaded.")

class VideoCamera(object):
    def __init__(self) -> None:
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frameV2(self):
        ret, frame = self.video.read()
        frame=cv2.resize(frame, None , fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)                    
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in face_rects:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            break
        # encode OpenCV raw frame to jpg and displaying it
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def get_frame(self):
        # Find haar cascade to draw bounding box around face
        ret, frame = self.video.read()
        frame = cv2.resize(frame, (1280, 720))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces available on camera
        num_faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        # take each face in view and preprocess it
        for (x, y, w, h) in num_faces:
            roi_frame = frame[y:y + h, x:x + w]
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_frame, (128, 128)), -1), 0)
            cropped_img_gray = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # predict whether wearing a mask
            maskPredict = maskModel.predict(cropped_img)
            maxindex = int(np.argmax(maskPredict))

            # use mask or unmasked set to recognize emotion depending on masked/unmasked
            if mask_dict[maxindex]=='Unmasked':
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 0, 255), 4)
                UnmaskedEmotionPredict = UnmaskedEmotion.predict(cropped_img_gray)
                maxindex2 = int(np.argmax(UnmaskedEmotionPredict))
                cv2.putText(frame, emotion_dict[maxindex2], (x + 200, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, mask_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                continue

            elif mask_dict[maxindex]=='Masked':
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
                MaskedEmotionPredict = MaskedEmotion.predict(cropped_img_gray)
                maxindex3 = int(np.argmax(MaskedEmotionPredict))
                cv2.putText(frame, emotion_dict[maxindex3], (x + 200, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, mask_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                continue
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()


