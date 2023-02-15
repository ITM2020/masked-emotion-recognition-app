import cv2
import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt

mask_dict = {0: "Masked", 1: "Unmasked"}
emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Surprised"}

# Load masked/unmasked model
json_file = open('Models/MaskModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
maskModel = model_from_json(loaded_model_json)
maskModel.load_weights("Models/MaskModel.h5")

# Load unmasked emotion recognition model
json_file = open('Models/EmotionUnmaskedModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
UnmaskedEmotion = model_from_json(loaded_model_json)
UnmaskedEmotion.load_weights("Models/EmotionUnmaskedModel.h5")

# Load masked emotion recognition model
json_file = open('Models/EmotionMaskedModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
MaskedEmotion = model_from_json(loaded_model_json)
MaskedEmotion.load_weights("Models/EmotionMaskedModel.h5")

print("Models have been loaded.")

# Load in the test image
test_img_unmasked = cv2.imread('Stills_Test/Nathan_Unmasked_Surprised.jpg')
test_img_unmasked_gray = cv2.cvtColor(test_img_unmasked, cv2.COLOR_BGR2GRAY)
test_img_masked = cv2.imread('Stills_Test/Nathan_Masked_Surprised.jpg')
test_img_masked_gray = cv2.cvtColor(test_img_masked, cv2.COLOR_BGR2GRAY)

# Initialize face detector object and detect faces
face_detector = cv2.CascadeClassifier('HaarCascadeFiles/haarcascade_frontalface_default.xml')
faces_unmasked = face_detector.detectMultiScale(test_img_unmasked_gray, scaleFactor=1.3, minNeighbors=5)
faces_masked = face_detector.detectMultiScale(test_img_masked_gray, scaleFactor=1.3, minNeighbors=5)

# Make prediction on unmasked faces
for (x, y, w, h) in faces_unmasked:
    # segment part of image containing face
    roi = test_img_unmasked[y:y + h, x:x + w]
    roi_gray = test_img_unmasked_gray[y:y + h, x:x + w]
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi, (128, 128)), -1), 0)
    cropped_img_gray = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

    # detect whether unmasked or masked
    maskPredict = maskModel.predict(cropped_img)
    maxindex = int(np.argmax(maskPredict))

    UnmaskedEmotionPredict = UnmaskedEmotion.predict(cropped_img_gray)
    maxindex2 = int(np.argmax(UnmaskedEmotionPredict))
    continue

# Make prediction on masked faces
for (x, y, w, h) in faces_masked:
    # segment part of image containing face
    roi = test_img_masked[y:y + h, x:x + w]
    roi_gray = test_img_masked_gray[y:y + h, x:x + w]
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi, (128, 128)), -1), 0)
    cropped_img_gray = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

    # detect whether unmasked or masked
    maskPredict = maskModel.predict(cropped_img)
    maxindex3 = int(np.argmax(maskPredict))

    UnmaskedEmotionPredict = UnmaskedEmotion.predict(cropped_img_gray)
    maxindex4 = int(np.argmax(UnmaskedEmotionPredict))

fig, (ax1, ax2) = plt.subplots(1,2)

fig.suptitle('Mask Detection & Emotion Recognition Results')
ax1.set_title("Mask Status: " + mask_dict[maxindex] + "\nEmotion Recognized: " + emotion_dict[maxindex2])
ax1.imshow(test_img_unmasked[:,:,::-1])
ax2.set_title("Mask Status: " + mask_dict[maxindex3] + "\nEmotion Recognized: " + emotion_dict[maxindex4])
ax2.imshow(test_img_masked[:,:,::-1])
plt.tight_layout()
plt.show()