from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay

emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Surprised"}

# Load masked/unmasked model
json_file = open('Models/EmotionUnmaskedModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
Unmasked_Emotion_Model = model_from_json(loaded_model_json)
Unmasked_Emotion_Model.load_weights("Models/EmotionUnmaskedModel.h5")

# Initialize image data generator with rescaling
test_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess all mask detection test images
test_generator_Unmasked_Emotion = test_data_gen.flow_from_directory(
        'Datasets/FER2013/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=False)

# do prediction on test data
predictions = Unmasked_Emotion_Model.predict_generator(test_generator_Unmasked_Emotion)

# confusion matrix
c_matrix = confusion_matrix(test_generator_Unmasked_Emotion.classes, predictions.argmax(axis=1))
print(c_matrix)
cm_display = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=['Angry', 'Happy', 'Neutral', 'Surprised'])
plt.figure(figsize=[8,6])
cm_display.plot(cmap=plt.cm.Blues)
plt.title('Unmasked Emotion Recognition Model: Confusion Matrix')
plt.show()
