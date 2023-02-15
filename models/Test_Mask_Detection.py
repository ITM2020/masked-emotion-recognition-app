from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay

mask_dict = {0: "Masked", 1: "Unmasked"}

# Load masked/unmasked model
json_file = open('Models/MaskModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
maskModel = model_from_json(loaded_model_json)
maskModel.load_weights("Models/MaskModel.h5")

# Initialize image data generator with rescaling
test_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess all mask detection test images
test_generator_Mask_Detection = test_data_gen.flow_from_directory(
        'Datasets/FaceMaskDataset/Test',
        target_size=(128, 128),
        batch_size=64,
        color_mode="rgb",
        class_mode='categorical',
        shuffle=False)

# do prediction on test data
predictions = maskModel.predict_generator(test_generator_Mask_Detection)

# confusion matrix
c_matrix = confusion_matrix(test_generator_Mask_Detection.classes, predictions.argmax(axis=1))
print(c_matrix)
cm_display = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=["Masked", "Unmasked"])
plt.figure(figsize=[8,6])
cm_display.plot(cmap=plt.cm.Blues)
plt.title('Mask Detection Model: Confusion Matrix', fontsize=16)
plt.show()