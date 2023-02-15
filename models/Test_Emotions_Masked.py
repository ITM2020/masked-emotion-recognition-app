from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay

emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Surprised"}

# Load masked/unmasked model
json_file = open('Models/EmotionMaskedModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
Masked_Emotion_Model = model_from_json(loaded_model_json)
Masked_Emotion_Model.load_weights("Models/EmotionMaskedModel.h5")

# Initialize image data generator with rescaling
test_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess all mask detection test images
test_generator_Masked_Emotion = test_data_gen.flow_from_directory(
        'Datasets/FER2013/test-masked',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=False)

# do prediction on test data
predictions = Masked_Emotion_Model.predict_generator(test_generator_Masked_Emotion)

# confusion matrix
c_matrix = confusion_matrix(test_generator_Masked_Emotion.classes, predictions.argmax(axis=1))
print(c_matrix)
cm_display = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=['Angry', 'Happy', 'Neutral', 'Surprised'])
plt.figure(figsize=[8,6])
cm_display.plot(cmap=plt.cm.Blues)
plt.title('Masked Emotion Recognition Model: Confusion Matrix')
plt.show()
