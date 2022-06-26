import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet169, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np;
import cv2
from keras.models import model_from_json
from pathlib import Path

faceCascade = cv2.CascadeClassifier('self-recognition/face-rec/src/haarcascade_frontalface_default.xml')

NAME = 'Mara'

classes = [NAME, 'Unknown']

# Load the pretrained CNN
densenet_model = DenseNet169(weights = 'imagenet',
                             include_top = False,
                             input_shape = (255, 255, 3))

# Load the json file that contains the model's structure
f = Path("self-recognition/face-rec/models/densenet169_model_structure.json")
model_structure = f.read_text()

# Recreate the Keras model object from the json data
model = model_from_json(model_structure)

# Re-load the model's trained weights
model.load_weights("self-recognition/face-rec/models/densenet169_model_weights.h5")

cap = cv2.VideoCapture(0)
cap.set(3,255) # set Width
cap.set(4,255) # set Height

def make_prediction(image):
    x = cv2.resize(img, (255,255))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    prediction = model.predict(x)
    
    most_likely_class_index = int(np.argmax(prediction[0]))
    label = classes[most_likely_class_index]
    
    return label

while True:
    ret, img = cap.read()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
        )

    for (x,y,w,h) in faces:
        roi_color = img[y:y+h, x:x+w]
        label = make_prediction(roi_color)
            
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        startY= y
        startX = x
        if startY - 20 > 20:
            y1 = startY - 20
            y2 = startY - 5
        else:
            y1 = startY + 30
            y2 = startY + 20
        cv2.rectangle(img, (startX, y1), (startX + w, startY), (0,0,255), -1)
        cv2.putText(img, label, (startX, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
    cv2.imshow('video',img)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()